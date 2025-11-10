# hello_video_gen.py
import os
import numpy as np
import torch
from pathlib import Path
from diffusers import DiffusionPipeline  # works for text-to-video models
import imageio

# Model note: CPU is very slow; GPU recommended.
MODEL_ID = os.environ.get("VIDEO_MODEL_ID", "damo-vilab/text-to-video-ms-1.7b")
PROMPT = os.environ.get("PROMPT", "a cat playing piano, cinematic lighting, 4k")
OUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "output_video.mp4"

print(f"[hello] Loading model: {MODEL_ID}")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
pipe = pipe.to(device)

# Fewer steps for demo; raise for better quality (slower)
NUM_STEPS = int(os.environ.get("NUM_STEPS", "25"))
FPS = int(os.environ.get("FPS", "8"))

print(f"[hello] Prompt: {PROMPT}")
print(f"[hello] Device: {device} | Steps: {NUM_STEPS} | FPS: {FPS}")

# Generate a short clip (list of PIL images)
result = pipe(PROMPT, num_inference_steps=NUM_STEPS)
frames = result.frames  # list of PIL Images

# Save as mp4
print(f"[hello] Writing {len(frames)} frames -> {OUT_PATH}")
with imageio.get_writer(OUT_PATH, fps=FPS) as writer:
    for im in frames:
        writer.append_data(np.array(im))

print(f"✅ Saved video: {OUT_PATH}")
