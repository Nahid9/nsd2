from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth
from tensorflow.keras.utils import custom_object_scope
import numpy as np
from PIL import Image
import io
import os

# Initialize FastAPI
app = FastAPI()

# Mount static files (CSS/JS if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (for HTML)
templates = Jinja2Templates(directory="static")

# Model config
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Leaf_Blight"]
# Pre-process class names to remove "Corn___" prefix and replace underscores with spaces
FORMATTED_CLASS_NAMES = [name[7:].replace("_", " ") for name in CLASS_NAMES]

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Load model with custom layers
custom_objects = {
    "RandomFlip": RandomFlip,
    "RandomRotation": RandomRotation,
    "RandomZoom": RandomZoom,
    "RandomHeight": RandomHeight,
    "RandomWidth": RandomWidth
}

with custom_object_scope(custom_objects):
    model = load_model("mobilenet_corn1.h5")  # Update path if needed

# Preprocess image
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img, axis=0)

# Homepage (HTML form)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        img = preprocess_image(io.BytesIO(image_bytes))
        
        # Predict
        preds = model.predict(img)
        class_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        predicted_class = FORMATTED_CLASS_NAMES[class_idx]  # Use the formatted class name
        
        # Return result as HTML
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": predicted_class,
                "confidence": f"{confidence:.2f}",
                "image_uploaded": True,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Error: {str(e)}"},
        )