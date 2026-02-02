import torch
from diffusers import StableDiffusionPipeline
import random

# -----------------------------
# Config – tweak these!
# -----------------------------
MODEL_ID = "CompVis/stable-diffusion-v1-4"  # Or "stabilityai/stable-diffusion-2-1" / "stabilityai/stable-diffusion-xl-base-1.0" for better quality (needs more VRAM)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available, else CPU (slow)
NUM_IMAGES = 1
STEPS = 50
GUIDANCE_SCALE = 7.5
SEED = random.randint(0, 1000000)

# Base prompt elements – mix & match for variety
anunnaki_themes = [
    "majestic Anunnaki god with golden wings, ancient Sumerian helmet, cybernetic enhancements, glowing quantum runes",
    "Anunnaki overlord in futuristic armor, holographic cuneiform floating around, quantum entanglement particles swirling",
    "ancient alien Anunnaki engineering human DNA in neon-lit quantum lab, wormholes and fractal energy fields",
    "Anunnaki warrior queen, blue skin, cyber implants, surrounded by quantum foam and plasma storms",
    "towering Anunnaki figure descending from Nibiru spaceship, futuristic quantum portal, neon cyberpunk ancient ruins"
]

futuristic_quantum_addons = [
    "neon cyberpunk style, quantum particles, entanglement effects, hard light holograms, octane render, 8k",
    "sci-fi dystopian, glowing energy fields, fractal geometry, cybernetic gold armor, dramatic lighting",
    "ultra-detailed, cinematic, quantum rift background, iridescent colors, ethereal glow, artstation trending"
]

negative_prompt = "blurry, lowres, deformed, ugly, bad anatomy, extra limbs, poorly drawn face, mutation"

# -----------------------------
# Load model (run once)
# -----------------------------
print("Loading Stable Diffusion model... (this may take a minute)")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()  # Saves VRAM

# -----------------------------
# Generate!
# -----------------------------
prompt = random.choice(anunnaki_themes) + ", " + random.choice(futuristic_quantum_addons) + ", highly detailed, masterpiece"

print(f"Generating with prompt: {prompt}")
print(f"Seed: {SEED}")

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

images = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=NUM_IMAGES,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=generator
).images

# Save
for i, img in enumerate(images):
    img.save(f"anunnaki_quantum_{SEED}_{i}.png")
    print(f"Saved: anunnaki_quantum_{SEED}_{i}.png")
