# app.py
# Final version, optimized for Railway's free tier with a memory-efficient model.

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
import cv2 # Ensure cv2 is imported for video processing

# --- Configuration for Railway's Persistent Volume ---
# This is where the AI model will be saved permanently after the first download.
CACHE_DIR = "/data/huggingface_cache"
# This is for temporary video files that are deleted after use.
GENERATED_DIR = "/tmp"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Global variable to hold the loaded AI model ---
ml_models = {}

# --- FastAPI Lifespan: Code to run on startup and shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the server starts up.
    It loads the AI model into memory.
    """
    # --- FINAL FIX: Using a smaller model that fits in Railway's 512MB RAM ---
    model_id = "segmind/tiny-sd"
    print(f"Loading AI model ({model_id}) from cache: {CACHE_DIR}")
    
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            variant="fp16", # Use the smaller fp16 variant for this model
            cache_dir=CACHE_DIR # Tell diffusers to use our persistent disk
        )
        pipe.to("cpu") # Ensure the model runs on the CPU
        ml_models["text_to_image"] = pipe
        print("✅ AI model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load AI model: {e}")
    
    yield # The application runs after this point
    
    # This code runs when the server shuts down
    ml_models.clear()
    print("AI model unloaded.")


# --- Initialize the FastAPI App ---
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic model for the request body ---
class VideoRequest(BaseModel):
    prompt: str

# --- Video Creation Function ---
def create_animated_video(base_image: Image.Image, output_path: str):
    """
    Takes a single image and creates a 5-second zoom-in video.
    """
    img_np = np.array(base_image)
    height, width, _ = img_np.shape
    num_frames = 5 * 24 # 5 seconds at 24 frames per second

    print("Creating video animation...")
    with imageio.get_writer(output_path, mode='I', fps=24, codec='libx264') as writer:
        for i in range(num_frames):
            # Create a gentle zoom-in effect
            scale = 1.0 + (0.2 * i / num_frames) 
            M = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
            zoomed_frame_np = cv2.warpAffine(img_np, M, (width, height))
            writer.append_data(zoomed_frame_np)
    print(f"✅ Animation saved to {output_path}")

# --- API Endpoints ---
@app.get("/")
def read_root():
    """ A simple endpoint to check if the server is running. """
    return {"message": "AI Video Generator API is running."}

@app.post("/generate-video")
async def generate_video_endpoint(request: VideoRequest):
    """ The main endpoint that generates and returns a video. """
    if "text_to_image" not in ml_models:
        raise HTTPException(status_code=503, detail="AI model is not available or is still loading.")
        
    print(f"Received prompt: '{request.prompt}'")
    try:
        # Step 1: Generate the base image from the text prompt
        pipe = ml_models["text_to_image"]
        # This smaller model needs more steps for good quality
        image = pipe(prompt=request.prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
        
        # Step 2: Animate the image into a video file
        video_path = os.path.join(GENERATED_DIR, f"{uuid.uuid4()}.mp4")
        create_animated_video(image, video_path)
        
        # Step 3: Read the video file and encode it in Base64 to send over the web
        with open(video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        # Step 4: Clean up the temporary file from the server
        os.remove(video_path)
        
        # Step 5: Return the video data to the frontend
        return {"video_base64": video_base64}
        
    except Exception as e:
        print(f"❌ Error during video generation: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred during video generation.")
