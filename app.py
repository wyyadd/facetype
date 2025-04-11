import gradio as gr

from main import get_pred_binary
from model import TARGET_LABELS


def get_face_type(img):
    try:
        pred_binary = get_pred_binary(img)
    except Exception as e:
        return str(e)

    result = "\n".join([f"{label}: {bool(pred)}" for label, pred in zip(TARGET_LABELS, pred_binary)])
    face_type = int(''.join(map(str, pred_binary)), 2)
    result = f"face_type: {face_type}\n{result}"
    return result


demo = gr.Interface(
    fn=get_face_type,
    inputs=["image"],
    outputs=["text"],
)

demo.launch()
