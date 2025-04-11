import cv2
import numpy as np
import requests
import torch
from deepface import DeepFace
from fastapi import FastAPI, HTTPException

from model import MultiLabelClassifier

app = FastAPI()

model_path = "classifier.pth"
model = MultiLabelClassifier(embedding_dim=4096, hidden_dim=1024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device).eval()


@app.get("/face-type")
def get_face_type(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses
        # Convert image data to numpy array in BGR format
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")

    pred_binary = get_pred_binary(img)

    face_type = int(''.join(map(str, pred_binary)), 2)

    return {"face_type": face_type}


def get_pred_binary(img: np.ndarray):
    try:
        embedding_objs = DeepFace.represent(
            img_path=img,
            model_name="VGG-Face")
    except Exception as e:
        raise HTTPException(status_code=500, detail="No face detected.")
    ebd = torch.tensor(embedding_objs[0]['embedding'], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(ebd)
        probs = torch.sigmoid(logits).cpu().numpy()
    pred_binary = (probs > 0.5).astype(int)

    return pred_binary
