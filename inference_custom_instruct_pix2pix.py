############## inference code #####################################################################
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

model_id = "/content/cartoonization-finetuned"

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=False
).to("cuda")

image_path = "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
image = load_image(image_path)

image = pipeline("Cartoonize the following image", image=image).images[0]
image.save("image.png")
######################################################################################################