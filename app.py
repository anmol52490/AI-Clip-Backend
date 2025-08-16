# app.py for Hugging Face Spaces
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
import imageio
import cv2
import tempfile
import os

# Load the smallest possible model
model_id = "OFA-Sys/small-stable-diffusion-v0"

print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

# Enable memory optimizations
try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass

pipe.enable_vae_slicing()
pipe.to("cpu")  # HF Spaces provides CPU by default

def create_simple_video(image, duration=3):
    """Create a simple zoom animation"""
    img_np = np.array(image)
    height, width = img_np.shape[:2]
    fps = 24
    num_frames = duration * fps
    
    # Create temporary video file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        video_path = tmp_file.name
    
    with imageio.get_writer(video_path, mode='I', fps=fps, codec='libx264') as writer:
        for i in range(num_frames):
            # Simple zoom effect
            scale = 1.0 + (0.2 * i / num_frames)
            M = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
            zoomed_frame = cv2.warpAffine(img_np, M, (width, height))
            writer.append_data(zoomed_frame)
    
    return video_path

def generate_video(prompt):
    """Generate image and convert to video"""
    try:
        # Generate image
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=15,  # Very few steps for speed
                guidance_scale=6.0,
                height=512,
                width=512
            ).images[0]
        
        # Create video
        video_path = create_simple_video(image, duration=3)
        
        return image, video_path
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Free AI Video Generator") as demo:
    gr.Markdown("# 🎬 Free AI Video Generator")
    gr.Markdown("Generate images and simple animated videos from text prompts using AI!")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="A beautiful sunset over mountains",
                lines=3
            )
            generate_btn = gr.Button("Generate Video", variant="primary")
        
        with gr.Column():
            image_output = gr.Image(label="Generated Image")
            video_output = gr.Video(label="Generated Video")
    
    generate_btn.click(
        generate_video,
        inputs=[prompt_input],
        outputs=[image_output, video_output]
    )
    
    gr.Examples(
        examples=[
            ["A cute cat playing in a garden"],
            ["A futuristic city skyline at night"],
            ["A peaceful lake with mountains in background"],
            ["A colorful flower bouquet"],
        ],
        inputs=[prompt_input]
    )

if __name__ == "__main__":
    demo.launch()
