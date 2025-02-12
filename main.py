import cv2
import numpy as np
import requests
from deepface import DeepFace
from fastapi import FastAPI, HTTPException

app = FastAPI()

np.random.seed(42)  # For reproducibility
hyperplanes = np.random.randn(512, 5)
# Optional: Normalize each hyperplane
hyperplanes /= np.linalg.norm(hyperplanes, axis=0)


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

    try:
        embedding_objs = DeepFace.represent(
            img_path=img,
            model_name="Facenet512")
    except Exception as e:
        raise HTTPException(status_code=500, detail="No face detected.")

    ebd = np.array(embedding_objs[0]['embedding'], dtype=np.float32)
    # Project vector onto hyperplanes
    projections = np.dot(ebd, hyperplanes)
    # Binarize (sign function)
    bits = (projections >= 0).astype(int)
    # Convert bits to integer (LSB first)
    face_type = int(''.join(map(str, bits)), 2)

    return {"face_type": face_type}

# def get_face_type(file):
#     try:
#         attribute = DeepFace.analyze(
#             img_path=file,
#             actions=['age', 'gender'],
#         )
#         gender = attribute[0]['dominant_gender']
#         age = attribute[0]['age']
#         if gender == 'Man':
#             if age < 10:
#                 face_type = 7
#             elif age < 20:
#                 face_type = 3
#             elif age < 30:
#                 face_type = 12
#             elif age < 40:
#                 face_type = 1
#             elif age < 50:
#                 face_type = 15
#             elif age < 60:
#                 face_type = 5
#             elif age < 70:
#                 face_type = 10
#             else:
#                 face_type = 8
#         elif gender == 'Woman':
#             if age < 10:
#                 face_type = 14
#             elif age < 20:
#                 face_type = 0
#             elif age < 30:
#                 face_type = 4
#             elif age < 40:
#                 face_type = 6
#             elif age < 50:
#                 face_type = 13
#             elif age < 60:
#                 face_type = 2
#             elif age < 70:
#                 face_type = 9
#             else:
#                 face_type = 11
#         else:
#             return "Face could not be detected."
#         return f"face type:{face_type}---gender:{gender}---age:{age}"
#     except Exception as e:
#         print(e)
#         return f"Face could not be detected."
#
#
# if __name__ == '__main__':
#     demo = gr.Interface(fn=get_new_face_type, inputs="image", outputs="label")
#     demo.launch(share=False)
