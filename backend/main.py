import os
import shutil
import base64
import json
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Import custom utilities (to be implemented)
from utils import Sam2Predictor, WhisperTranscriber, QwenVLGenerator

from fastapi.staticfiles import StaticFiles

from starlette.requests import Request
import time

app = FastAPI()

# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    print(f"[ACCESS] Incoming request: {method} {path}")
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        print(f"[ACCESS] Completed: {method} {path} - Status: {response.status_code} - Time: {process_time:.2f}ms")
        return response
    except Exception as e:
        print(f"[ACCESS ERROR] Request failed: {method} {path} - Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e

# Ensure temp directory exists before mounting
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/healthz")
async def health_check():
    return {"status": "ok", "transformers": os.environ.get("TRANSFORMERS_VERSION", "unknown")}

# Mount temp directory for static access
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize models (lazy loading or startup)
sam2_predictor = None
whisper_transcriber = None
qwen_vl_generator = None

@app.on_event("startup")
async def startup_event():
    global sam2_predictor, whisper_transcriber, qwen_vl_generator
    print("Loading models...")
    # Initialize your models here
    # sam2_predictor = Sam2Predictor()
    # whisper_transcriber = WhisperTranscriber()
    # qwen_vl_generator = QwenVLGenerator()
    print("Models loaded (placeholders active).")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": file_path}

@app.post("/predict")
async def predict(
    video_path: str = Form(...),
    x: Optional[float] = Form(None),
    y: Optional[float] = Form(None),
    points_json: Optional[str] = Form(None),
    labels_json: Optional[str] = Form(None),
    timestamp: float = Form(...),  # Time in seconds
    frame_width: int = Form(...),
    frame_height: int = Form(...),
    api_key: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    qwen_model: str = Form("Qwen/Qwen2-VL-7B-Instruct"),
    sam2_model: str = Form("facebook/sam2-hiera-tiny")
):
    global sam2_predictor, whisper_transcriber, qwen_vl_generator
    
    try:
        # 1. Initialize models if needed
        if sam2_predictor is None:
             sam2_predictor = Sam2Predictor(model_id=sam2_model)
        elif sam2_predictor.model_id != sam2_model:
             print(f"Switching SAM2 model from {sam2_predictor.model_id} to {sam2_model}")
             sam2_predictor = Sam2Predictor(model_id=sam2_model)

        if whisper_transcriber is None:
             whisper_transcriber = WhisperTranscriber()
        if qwen_vl_generator is None:
             qwen_vl_generator = QwenVLGenerator()

        # 2. Extract Frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=400, detail="Could not read frame")

        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / frame_width
        scale_y = orig_h / frame_height
        
        final_points = []
        final_labels = []

        # Process new points/labels format
        if points_json and labels_json:
            try:
                raw_points = json.loads(points_json)
                raw_labels = json.loads(labels_json)
                
                if len(raw_points) != len(raw_labels):
                    raise ValueError("Points and labels length mismatch")
                
                # Optimization: Downsample points if too many (e.g. scribble)
                # SAM2 works best with fewer, well-placed points.
                # If we have > 30 points, we take every K-th point.
                if len(raw_points) > 30:
                    step = len(raw_points) // 20
                    raw_points = raw_points[::step]
                    raw_labels = raw_labels[::step]
                    print(f"[Main] Downsampled points from {len(raw_points)*step} to {len(raw_points)}")

                for p in raw_points:
                    final_points.append([int(p[0] * scale_x), int(p[1] * scale_y)])
                final_labels = raw_labels
                
                print(f"[Main] Received {len(final_points)} points via JSON.")
            except Exception as e:
                print(f"[Main] Error parsing points_json/labels_json: {e}")
        
        # Fallback to single point if no list provided
        if not final_points and x is not None and y is not None:
            actual_x = int(x * scale_x)
            actual_y = int(y * scale_y)
            final_points = [[actual_x, actual_y]]
            final_labels = [1]
            print(f"[Main] Using legacy single point: ({actual_x}, {actual_y})")
            
        if not final_points:
            raise HTTPException(status_code=400, detail="No points provided")
        
        print(f"--- Processing Start ---")
        
        # 3. SAM2 Inference
        print(f"[Main] Step 2: Running SAM2 Video Segmentation with {len(final_points)} points...")
        # Old single frame call:
        # mask, masked_image = sam2_predictor.predict(frame, final_points, final_labels)
        
        # New video call:
        output_video_path = sam2_predictor.predict_video(video_path, final_points, final_labels, timestamp)
        
        # We still need a single mask for QwenVL?
        # QwenVL usually describes the object. 
        # Let's use the mask from the *prompt frame* for QwenVL generation.
        # So we can call predict() once for the prompt frame to get the mask/image for Qwen.
        mask, masked_image = sam2_predictor.predict(frame, final_points, final_labels)
        
        # 4. Whisper Transcription
        print(f"[Main] Step 3: Running Whisper Transcription at {timestamp}s...")
        audio_text = whisper_transcriber.transcribe_segment(video_path, timestamp, duration=5.0)
        print(f"[Main] Whisper Result: {audio_text}")
        
        # 5. Qwen VL Generation
        print(f"[Main] Step 4: Running Qwen VL Encyclopedia Generation...")
        encyclopedia_text = qwen_vl_generator.generate(
            masked_image, 
            audio_text, 
            api_key=api_key, 
            base_url=base_url,
            model_name=qwen_model
        )
        print(f"[Main] Qwen Result: {encyclopedia_text[:100]}...")
        print(f"--- Processing End ---")
        
        # 6. Encode mask for response
        # We return the video path now.
        # But frontend expects "mask" as base64 image.
        # We should update frontend to accept video url.
        # For backward compatibility or immediate display, we can still return the frame mask.
        # And ADD the video url.
        
        _, buffer = cv2.imencode('.png', (mask * 255).astype(np.uint8))
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create a temp URL for the output video
        output_filename = os.path.basename(output_video_path)
        video_url = f"http://localhost:8000/temp/{output_filename}"
        
        return JSONResponse({
            "mask": f"data:image/png;base64,{mask_base64}", # Keep for Qwen logic/compatibility
            "transcription": audio_text,
            "encyclopedia": encyclopedia_text,
            "segmented_video_url": video_url # New field
        })

    except Exception as e:
        import traceback
        error_msg = f"Prediction failed: {str(e)}"
        print(f"[CRITICAL ERROR] {error_msg}")
        print(traceback.format_exc())
        # Return 500 with details for debugging
        return JSONResponse(
            status_code=500,
            content={
                "detail": error_msg,
                "traceback": traceback.format_exc()
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
