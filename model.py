import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas
import pandas as pd
import torch
from deepface import DeepFace
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

TARGET_LABELS = ["Male", "Young", "Oval_Face", "High_Cheekbones", "Big_Lips", "Big_Nose"]


def load_df(target_labels: list[str]):
    # 1. load CSV file
    partition_df = pd.read_csv('./data/list_eval_partition.csv')
    labels_df = pd.read_csv('./data/list_attr_celeba.csv')

    # 2. merge two tables
    df = pd.merge(partition_df, labels_df, on='image_id')

    # 3. mapping label: -1 -> 0
    for label in target_labels:
        df[label] = (df[label] + 1) // 2  # 转成 0/1

    # 4. subset
    train_df = df[df['partition'] != 2]
    test_df = df[df['partition'] == 2]

    return train_df, test_df


class EmbeddingDataset(Dataset):
    def __init__(self, df: pandas.DataFrame, target_labels: list[str]):
        self.df = df
        self.image_root = Path("./data/img_align_celeba/img_align_celeba/")
        self.target_labels = target_labels
        self.preprocess()

    def preprocess(self):
        to_process_images = [image_id for image_id in self.df['image_id'] if
                             not (self.image_root / f"{image_id}.pkl").exists()]
        if len(to_process_images) > 0:
            logging.info(f"Preprocessing {len(to_process_images)} images")
        else:
            return
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._process_image, image_id) for image_id in to_process_images]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing image: {e}")

    def _process_image(self, image_id: str):
        # Get the image path and cache file path
        image_path = self.image_root / image_id
        cache_file = self.image_root / f"{image_id}.pkl"

        # Check if the embedding is already cached
        if not cache_file.exists():
            # Generate the embedding if it is not cached
            embedding_obj = DeepFace.represent(
                img_path=str(image_path),
                model_name="VGG-Face",
                enforce_detection=False
            )
            embedding = torch.tensor(embedding_obj[0]["embedding"], dtype=torch.float32)

            # Save the embedding to a pickle file for future use
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get embedding
        cache_file = self.image_root / f"{row['image_id']}.pkl"
        with open(cache_file, "rb") as f:
            embedding = pickle.load(f)

        # Get labels
        labels = torch.from_numpy(row[self.target_labels].values.astype(int))
        return embedding, labels


class MultiLabelClassifier(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = len(TARGET_LABELS)
        self.dropout = 0.1
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, len(TARGET_LABELS)),
        )

    def forward(self, x):
        return self.classifier(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy(probs, targets.float(), reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("train.log"),
            logging.StreamHandler()  # Also log to the console
        ]
    )
    train_df, test_df = load_df(TARGET_LABELS)
    # filter df
    # train_df, test_df = train_df[train_df.index % 5 == 0], test_df[test_df.index % 5 == 0]
    train_dataset = EmbeddingDataset(train_df, TARGET_LABELS)
    test_dataset = EmbeddingDataset(test_df, TARGET_LABELS)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32)
    logging.info(f"Initializing Dataset, train_loader: {len(train_loader)}, test_loader: {len(test_loader)}")

    device = torch.device("mps")
    logging.info(f"Using device: {device}")

    model = MultiLabelClassifier(embedding_dim=4096, hidden_dim=1024).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    logging.info("Initializing model, optimizer and criterion")
    logging.info("Starting training")

    for epoch in range(50):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if epoch % 5 == 0:
            model.eval()
            test_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for inputs, targets in tqdm(test_loader, desc=f"Test Epoch {epoch}"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.float())

                    test_loss += loss.item()
                    predicted = torch.sigmoid(outputs) > 0.5
                    all_preds.append(predicted)
                    all_targets.append(targets)

            avg_test_loss = test_loss / len(test_loader)
            all_preds = torch.cat(all_preds).cpu().numpy()
            all_targets = torch.cat(all_targets).cpu().numpy()

            accuracy = accuracy_score(all_targets, all_preds)
            recall = recall_score(all_targets, all_preds, average='macro')
            f1 = f1_score(all_targets, all_preds, average='macro')

            logging.info(
                f"Epoch {epoch} - Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    torch.save(model.state_dict(), "data/classifier.pth")


if __name__ == "__main__":
    main()
