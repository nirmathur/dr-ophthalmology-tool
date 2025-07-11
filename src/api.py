from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model

from src import config
from src.evaluate import make_gradcam_heatmap, get_last_conv_layer_name

app = FastAPI(title="DR Model API")


@app.on_event("startup")
def load_trained_model():
    global model, last_conv
    model = load_model(config.MODEL_SAVE_PATH)
    last_conv = get_last_conv_layer_name(model)


def apply_gradcam_overlay(img_rgb, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, config.IMAGE_SIZE)
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    grade = int(np.argmax(preds[0]))
    heatmap = make_gradcam_heatmap(img_array, model, last_conv, grade)
    overlay = apply_gradcam_overlay(img_rgb, heatmap)
    _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buf).decode()
    return JSONResponse({"grade": grade, "overlay": overlay_b64})

