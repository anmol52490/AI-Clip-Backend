# app.py
# Final version, optimized for Vercel's serverless environment.

import os
import uuid
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from diffusers import AutoPipelineForText2Image
import numpy as np
from PIL import Image
import imageio
import cv2

# --- Configuration for Vercel's Environment ---
# Vercel provides a single writable directory: /tmp. We'll use it for everything.
CACHE_DIR = "/tmp/huggingface_cache"
GENERATED_DIR = "/tmp"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Global variable to hold the loaded AI model ---
ml_models = {}

# --- FastAPI Lifespan: Code to run on startup and shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the serverless function starts up.
    It loads the AI model into memory from Vercel's temporary storage.
    """
    model_id = "segmind/tiny-sd"
    print(f"Loading AI model ({model_id}) from cache: {CACHE_DIR}")
    
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            # variant="fp16",
            cache_dir=CACHE_DIR # Tell diffusers to use the /tmp directory
        )
        pipe.to("cpu")
        ml_models["text_to_image"] = pipe
        print("✅ AI model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load AI model: {e}")
    
    yield # The application runs after this point
    
    ml_models.clear()
    print("AI model unloaded.")


# --- Initialize the FastAPI App ---
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow the frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    prompt: str

def create_animated_video(base_image: Image.Image, output_path: str):
    img_np = np.array(base_image)
    height, width, _ = img_np.shape
    num_frames = 5 * 24
    with imageio.get_writer(output_path, mode='I', fps=24, codec='libx264') as writer:
        for i in range(num_frames):
            scale = 1.0 + (0.2 * i / num_frames)
            M = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
            zoomed_frame_np = cv2.warpAffine(img_np, M, (width, height))
            writer.append_data(zoomed_frame_np)

@app.get("/")
def read_root():
    return {"message": "This is the root of the API. It's meant to be called by a frontend."}

@app.post("/generate-video")
async def generate_video_endpoint(request: VideoRequest):
    if "text_to_image" not in ml_models:
        raise HTTPException(status_code=503, detail="AI model is not available or is still loading.")
    try:
        pipe = ml_models["text_to_image"]
        image = pipe(prompt=request.prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
        video_path = os.path.join(GENERATED_DIR, f"{uuid.uuid4()}.mp4")
        create_animated_video(image, video_path)
        with open(video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        os.remove(video_path)
        return {"video_base64": video_base64}
    except Exception as e:
        print(f"❌ Error during video generation: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
