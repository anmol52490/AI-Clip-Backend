# app.py
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

# --- Configuration for Railway's Persistent Volume ---
CACHE_DIR = "/data/huggingface_cache"
GENERATED_DIR = "/tmp"
os.makedirs(CACHE_DIR, exist_ok=True)

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_id = "stabilityai/sd-turbo"
    print(f"Loading AI model ({model_id}) from cache: {CACHE_DIR}")
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            cache_dir=CACHE_DIR
        )
        pipe.to("cpu")
        ml_models["text_to_image"] = pipe
        print("✅ AI model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load AI model: {e}")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

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
    num_frames = 5 * 24 # 5 seconds at 24 fps
    with imageio.get_writer(output_path, mode='I', fps=24, codec='libx264') as writer:
        for i in range(num_frames):
            scale = 1.0 + (0.2 * i / num_frames)
            M = cv2.getRotationMatrix2D((img_np.shape[1]/2, img_np.shape[0]/2), 0, scale)
            zoomed_frame_np = cv2.warpAffine(img_np, M, (img_np.shape[1], img_np.shape[0]))
            writer.append_data(zoomed_frame_np)

@app.get("/")
def read_root():
    return {"message": "AI Video Generator API is running."}

@app.post("/generate-video")
async def generate_video_endpoint(request: VideoRequest):
    if "text_to_image" not in ml_models:
        raise HTTPException(status_code=503, detail="AI model is not available.")
    try:
        pipe = ml_models["text_to_image"]
        image = pipe(prompt=request.prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        
        video_path = os.path.join(GENERATED_DIR, f"{uuid.uuid4()}.mp4")
        create_animated_video(image, video_path)
        
        with open(video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        os.remove(video_path)
        return {"video_base64": video_base64}
    except Exception as e:
        print(f"❌ Error during video generation: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
