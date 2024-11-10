from diffusers import DiffusionPipeline
from huggingface_hub import login
import os
os.environ["TRANSFORMERS_CACHE"] = "./stablediff"
os.environ["DIFFUSERS_CACHE"] = "./stablediff"
os.environ["HF_HOME"] = "./stablediff"
os.environ["HF_HUB_CACHE"] = "./stablediff"

login(token="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")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]
image.save(astronaut.png)
