import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
import imageio
import cv2
import tempfile
import os
import gc

# Check if CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the most memory-efficient model
model_id = "OFA-Sys/small-stable-diffusion-v0"

print("🚀 Loading AI model... This may take a moment on first run.")

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True
    )
    
    # Move to appropriate device
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xformers not available, using default attention")
    
    pipe.enable_vae_slicing()
    
    # Additional memory optimizations for CPU
    if device == "cpu":
        try:
            pipe.enable_sequential_cpu_offload()
        except:
            pass
    
    print("✅ Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Fallback to a different model if the first fails
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        pipe.enable_vae_slicing()
        print("✅ Fallback model loaded successfully!")
    except Exception as fallback_error:
        print(f"❌ Fallback model also failed: {fallback_error}")
        pipe = None

def create_animated_video(image, duration=4, animation_type="zoom"):
    """Create different types of animations from a static image"""
    img_np = np.array(image)
    height, width = img_np.shape[:2]
    fps = 24
    num_frames = duration * fps
    
    # Create temporary video file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        video_path = tmp_file.name
    
    try:
        with imageio.get_writer(video_path, mode='I', fps=fps, codec='libx264', quality=8) as writer:
            for i in range(num_frames):
                progress = i / num_frames
                
                if animation_type == "zoom":
                    # Zoom in effect
                    scale = 1.0 + (0.3 * progress)
                    M = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
                    frame = cv2.warpAffine(img_np, M, (width, height))
                
                elif animation_type == "pan":
                    # Pan left to right
                    shift_x = int(50 * np.sin(progress * np.pi))
                    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
                    frame = cv2.warpAffine(img_np, M, (width, height))
                
                elif animation_type == "rotate":
                    # Gentle rotation
                    angle = 15 * np.sin(progress * 2 * np.pi)
                    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
                    frame = cv2.warpAffine(img_np, M, (width, height))
                
                else:  # fade
                    # Fade in and out
                    alpha = 0.5 + 0.5 * np.sin(progress * 2 * np.pi)
                    frame = (img_np * alpha).astype(np.uint8)
                
                writer.append_data(frame)
        
        return video_path
    
    except Exception as e:
        print(f"Error creating video: {e}")
        return None

def generate_content(prompt, animation_type, duration, num_steps, guidance_scale):
    """Generate image and video from text prompt"""
    if pipe is None:
        return None, None, "❌ Model failed to load. Please try refreshing the page."
    
    if not prompt.strip():
        return None, None, "⚠️ Please enter a prompt!"
    
    try:
        # Update status
        status = "🎨 Generating image..."
        
        # Generate image
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            )
            image = result.images[0]
        
        # Force garbage collection
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        status = "🎬 Creating animation..."
        
        # Create video
        video_path = create_animated_video(image, duration, animation_type)
        
        if video_path is None:
            return image, None, "✅ Image generated! ❌ Video creation failed."
        
        return image, video_path, f"✅ Complete! Generated with {num_steps} steps."
        
    except Exception as e:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return None, None, f"❌ Error: {str(e)}"

def generate_image_only(prompt, num_steps, guidance_scale):
    """Generate only image for faster testing"""
    if pipe is None:
        return None, "❌ Model failed to load."
    
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt!"
    
    try:
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            )
            image = result.images[0]
        
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return image, f"✅ Image generated with {num_steps} steps!"
        
    except Exception as e:
        gc.collect()
        return None, f"❌ Error: {str(e)}"

# Custom CSS for better appearance
css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.main-header {
    text-align: center;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 1em;
}
.status-box {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
"""

# Create the Gradio interface
with gr.Blocks(css=css, title="🎬 Free AI Video Generator", theme=gr.themes.Soft()) as demo:
    
    gr.HTML('<h1 class="main-header">🎬 AI Video Generator</h1>')
    gr.Markdown(
        """
        ### 🚀 Generate Amazing Videos from Text!
        Create stunning images and animated videos using AI. Perfect for social media, presentations, or creative projects!
        
        **💡 Tips for better results:**
        - Be descriptive: "A serene mountain lake at golden hour with mist"
        - Specify style: "...in anime style" or "...photorealistic"
        - Add mood: "peaceful", "dramatic", "vibrant"
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            prompt_input = gr.Textbox(
                label="✍️ Enter your prompt",
                placeholder="A majestic dragon flying over a medieval castle at sunset",
                lines=3,
                max_lines=5
            )
            
            # Advanced settings
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    num_steps = gr.Slider(
                        label="Quality (steps)",
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=5,
                        info="More steps = better quality but slower"
                    )
                    guidance_scale = gr.Slider(
                        label="Prompt Strength",
                        minimum=3,
                        maximum=15,
                        value=7.5,
                        step=0.5,
                        info="How closely to follow the prompt"
                    )
                
                with gr.Row():
                    animation_type = gr.Dropdown(
                        label="Animation Type",
                        choices=["zoom", "pan", "rotate", "fade"],
                        value="zoom",
                        info="Type of animation for the video"
                    )
                    duration = gr.Slider(
                        label="Video Duration (seconds)",
                        minimum=2,
                        maximum=8,
                        value=4,
                        step=1
                    )
            
            # Buttons
            with gr.Row():
                generate_video_btn = gr.Button(
                    "🎬 Generate Video", 
                    variant="primary",
                    size="lg"
                )
                generate_image_btn = gr.Button(
                    "🖼️ Image Only (Fast)",
                    variant="secondary"
                )
        
        with gr.Column(scale=3):
            # Output section
            status_output = gr.Textbox(
                label="📊 Status",
                interactive=False,
                max_lines=2
            )
            
            image_output = gr.Image(
                label="🖼️ Generated Image",
                show_download_button=True,
                height=400
            )
            
            video_output = gr.Video(
                label="🎬 Generated Video",
                show_download_button=True,
                height=400
            )
    
    # Example prompts
    gr.Markdown("### 🎯 Try These Example Prompts:")
    
    examples = [
        ["A cute corgi puppy playing in a field of sunflowers", "zoom", 4, 25, 7.5],
        ["A futuristic cyberpunk city with neon lights at night", "pan", 5, 30, 8.0],
        ["A magical forest with glowing mushrooms and fairy lights", "rotate", 4, 25, 7.5],
        ["A serene Japanese garden with cherry blossoms", "fade", 3, 20, 7.0],
        ["A space station orbiting Earth with stars in background", "zoom", 6, 35, 8.5],
        ["A cozy cabin by a lake during autumn", "pan", 4, 25, 7.5]
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[prompt_input, animation_type, duration, num_steps, guidance_scale],
        label="Click any example to load it"
    )
    
    # Event handlers
    generate_video_btn.click(
        fn=generate_content,
        inputs=[prompt_input, animation_type, duration, num_steps, guidance_scale],
        outputs=[image_output, video_output, status_output],
        show_progress=True
    )
    
    generate_image_btn.click(
        fn=generate_image_only,
        inputs=[prompt_input, num_steps, guidance_scale],
        outputs=[image_output, status_output],
        show_progress=True
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        ### 💻 System Info
        - **Model**: OFA Small Stable Diffusion (Memory Optimized)
        - **Device**: """ + f"{device.upper()}" + """
        - **Platform**: Hugging Face Spaces (Free Tier)
        
        **🔄 First generation may take longer due to model loading.**
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.queue(max_size=10)  # Enable queuing for multiple users
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False  # Set to True if you want a public link
    )
