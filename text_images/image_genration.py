from huggingface_hub import login
from diffusers import StableDiffusionXLPipeline
import torch
import os 
from PIL import Image
import numpy as np

# Initialize the Diffuser pipeline and configuration
login()
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")
config = pipe.unet.config


def process_image(prompt: str, save_dir: str):
    if not prompt:
        return {"error": "Prompt is missing"}

    # Generate the image
    image = pipe(prompt=prompt, num_inference_steps=30).images[0]

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path where the image will be saved
    file_name = "generated_image.png"  # You can use a different file name and extension if needed
    save_path = os.path.join(save_dir, file_name)

    # Convert the generated image tensor to a NumPy array
    image_array = np.array(image).astype('uint8')

    # Create a PIL image from the NumPy array
    generated_image = Image.fromarray(image_array)

    # Save the PIL image to the specified file path
    generated_image.save(save_path)

    # Optional: Display the generated image
    generated_image.show()

    # Return the file path
    return {"file_path": save_path}

# You can add more functions and code related to image generation here
