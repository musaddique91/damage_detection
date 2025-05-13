from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from typing import List
from ultralytics import YOLO
import cv2
import uuid
import os

app = FastAPI()

# Load model once at startup
model = YOLO("best.pt")

# Make sure output folder exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/detect/")
async def detect_damage(files: List[UploadFile] = File(...)):
    results_list = []

    for file in files:
        # Generate unique filename
        file_ext = file.filename.split('.')[-1]
        temp_filename = f"{uuid.uuid4()}.{file_ext}"
        temp_path = os.path.join(OUTPUT_DIR, temp_filename)

        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Run detection
        results = model(temp_path, save=True, project=OUTPUT_DIR, name='runs', exist_ok=True)

        # Get the saved image path (Ultralytics saves to runs directory)
        saved_image_path = os.path.join(OUTPUT_DIR, 'runs', os.path.basename(temp_path))

        if os.path.exists(saved_image_path):
            results_list.append(saved_image_path)
        else:
            results_list.append(temp_path)  # fallback

    # If only one image, return it directly
    if len(results_list) == 1:
        return FileResponse(results_list[0], media_type="image/jpeg", filename="result.jpg")

    # If multiple, return ZIP or filenames (optional – for now just names)
    return {"processed_files": [os.path.basename(p) for p in results_list]}
