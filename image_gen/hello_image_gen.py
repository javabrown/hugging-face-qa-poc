# hello_image_gen.py
from diffusers import StableDiffusionPipeline
from pathlib import Path
import os, torch

model_id = "runwayml/stable-diffusion-v1-5"
prompt = os.environ.get("PROMPT", "A boy with rose")
out_dir = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
out_dir.mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")

img = pipe(prompt, num_inference_steps=15, guidance_scale=7.5).images[0]
out_path = out_dir / "output.png"
img.save(out_path)

print(f"✅ Saved: {out_path}")
