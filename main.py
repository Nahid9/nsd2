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
import requests
from pathlib import Path

# Initialize FastAPI
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# --- Google Drive Configuration --- #
DRIVE_FILE_ID = "1hPsBx3fZTHN4VAmEoQd8ZQOSr7Emh_za"  # From your shareable link
MODEL_NAME = "mobilenet_corn1.h5"
MODEL_PATH = f"/tmp/{MODEL_NAME}"  # Vercel's temporary directory

def download_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        # Direct download URL with cookie handling
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        
        session = requests.Session()
        
        # First request to get confirmation token for large files
        response = session.get(url, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # Second request with confirmation token if needed
        if token:
            url = f"{url}&confirm={token}"
            response = session.get(url, stream=True)
        
        # Save in chunks to handle large files
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        print(f"Model saved to {MODEL_PATH}")
    else:
        print("Model already exists")

# --- Model Configuration --- #
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Leaf_Blight"]
FORMATTED_CLASS_NAMES = [name[7:].replace("_", " ") for name in CLASS_NAMES]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Load model at startup
@app.on_event("startup")
async def load_model():
    try:
        download_model_from_drive()
        
        custom_objects = {
            "RandomFlip": RandomFlip,
            "RandomRotation": RandomRotation,
            "RandomZoom": RandomZoom,
            "RandomHeight": RandomHeight,
            "RandomWidth": RandomWidth
        }
        
        with custom_object_scope(custom_objects):
            app.state.model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Create a dummy model if real one fails (for testing)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        app.state.model = Sequential([Dense(1)])
        print("Created dummy model for fallback")

# --- Image Processing and Routes --- #
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img, axis=0)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        if not hasattr(app.state, 'model'):
            raise RuntimeError("Model not loaded")
            
        image_bytes = await file.read()
        img = preprocess_image(io.BytesIO(image_bytes))
        
        preds = app.state.model.predict(img)
        class_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        predicted_class = FORMATTED_CLASS_NAMES[class_idx]
        
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
