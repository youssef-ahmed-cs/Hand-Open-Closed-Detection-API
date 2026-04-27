import io
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI(
    title="Hand Open/Closed Detection API",
    description="Detects whether a hand is open or closed from an uploaded image using MediaPipe Hand Landmarker.",
    version="1.0.0",
    docs_url=None,
    openapi_url=None,
    redoc_url=None,
)

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
)

landmarker = vision.HandLandmarker.create_from_options(options)

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIP_JOINTS = [6, 10, 14, 18]


def classify_hand(landmarks: list) -> dict:
    opened_fingers = 0

    for tip_index, pip_index in zip(FINGER_TIPS, FINGER_PIP_JOINTS):
        if landmarks[tip_index].y < landmarks[pip_index].y:
            opened_fingers += 1

    thumb_open = False
    if abs(landmarks[4].x - landmarks[5].x) > 0.05:
        thumb_open = landmarks[4].y < landmarks[3].y
    if thumb_open:
        opened_fingers += 1

    if opened_fingers <= 1:
        status = "Hand Closed"
    elif opened_fingers >= 3:
        status = "Hand Open"
    else:
        status = "Unclear"

    return {
        "status": status,
        "opened_fingers": opened_fingers,
        "finger_tips_detected": len(FINGER_TIPS) + (1 if thumb_open else 0),
    }


@app.post("/detect")
async def detect_hand(image: UploadFile = File(...)):
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    image_bytes = await image.read()
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode the uploaded image.")

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = landmarker.detect(mp_image)

    if not results.hand_landmarks:
        return JSONResponse({"status": "No hand detected", "opened_fingers": 0})

    landmarks = results.hand_landmarks[0]
    classification = classify_hand(landmarks)
    return JSONResponse(classification)


@app.get("/", response_class=FileResponse)
def serve_frontend():
    return str(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health_check():
    return {"message": "FastAPI Hand Open/Closed Detection is running."}


# Mount static files after routes to avoid conflicts
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import uvicorn # serve the app using uvicorn ASGI server
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)