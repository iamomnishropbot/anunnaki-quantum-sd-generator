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

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import argparse
import random
from torchvision.transforms import functional as TF
import cv2  # For canny edge detection

# -----------------------------
# CLI Args for Advanced Control
# -----------------------------
parser = argparse.ArgumentParser(description="Advanced Anunnaki Quantum SDXL Generator")
parser.add_argument("--prompt", type=str, default=None, help="Custom prompt (overrides random)")
parser.add_argument("--num_images", type=int, default=2, help="Number of images to generate")
parser.add_argument("--steps", type=int, default=60, help="Inference steps (higher = better quality)")
parser.add_argument("--guidance", type=float, default=8.5, help="Guidance scale")
parser.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducibility")
parser.add_argument("--use_controlnet", action="store_true", help="Enable ControlNet (canny edges)")
parser.add_argument("--control_image", type=str, default=None, help="Path to control image for canny")
parser.add_argument("--use_img2img", action="store_true", help="Enable img2img mode")
parser.add_argument("--init_image", type=str, default=None, help="Init image for img2img")
parser.add_argument("--strength", type=float, default=0.75, help="Img2img strength (0-1)")
parser.add_argument("--upscale", action="store_true", help="Upscale output 2x")
args = parser.parse_args()

# -----------------------------
# Advanced Config
# -----------------------------
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
CONTROLNET_ID = "diffusers/controlnet-canny-sdxl-1.0"  # Advanced ControlNet for SDXL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SEED = args.seed or random.randint(0, 2**32 - 1)

# Expanded Prompt Banks with Weighting
anunnaki_bases = [
    "(majestic Anunnaki overlord with golden cybernetic wings:1.2), ancient Sumerian helmet fused with quantum neural implants",
    "(blue-skinned Anunnaki genetic engineer:1.1), manipulating holographic DNA strands in a neon quantum chamber",
    "(towering Anunnaki warrior queen:1.3), clad in iridescent cyber armor, wielding plasma-infused cuneiform staff",
    "(Anunnaki council in zero-gravity throne room:1.0), surrounded by entangled quantum particles and wormhole portals",
    "(ancient alien Anunnaki descending from Nibiru mothership:1.2), with fractal energy auras and holographic runes"
]

quantum_futuristic = [
    ", neon cyberpunk quantum entanglement fields (swirling particles:1.4), hard light holograms, octane render, 8k ultra-detailed",
    ", dystopian sci-fi, glowing fractal geometry (cybernetic gold enhancements:1.2), dramatic volumetric lighting, cinematic",
    ", ethereal quantum rift background, iridescent plasma colors (entangled dimensions:1.3), artstation trending, masterpiece",
    ", futuristic ancient ruins under aurora borealis, cyber-implants glowing, quantum foam bubbling (high-tech Sumerian artifacts:1.1)"
]

negative_prompt = "blurry, lowres, deformed, mutated, ugly, disfigured, poorly drawn face, bad anatomy, extra limb, missing limb, floating limbs, disconnected limbs, malformed hands, text, watermark, overexposed, underexposed"

# -----------------------------
# Load Pipelines (Advanced: Base + Refiner + ControlNet)
# -----------------------------
print("Loading SDXL base + refiner...")
base_pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, variant="fp16" if DTYPE == torch.float16 else None)
base_pipe.to(DEVICE)
base_pipe.enable_attention_slicing()

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(REFINER_ID, torch_dtype=DTYPE, variant="fp16" if DTYPE == torch.float16 else None)
refiner.to(DEVICE)

if args.use_controlnet:
    print("Loading ControlNet (canny)...")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=DTYPE)
    pipe = StableDiffusionXLControlNetPipeline.from_pipe(base_pipe, controlnet=controlnet)
else:
    pipe = base_pipe

if args.use_img2img:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)

# -----------------------------
# Prep Control Image (If Enabled)
# -----------------------------
control_image = None
if args.use_controlnet and args.control_image:
    init_img = load_image(args.control_image).convert("RGB")
    init_img = np.array(init_img)
    # Canny edge detection for advanced guidance
    gray = cv2.cvtColor(init_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.dilate(edges, None)
    control_image = Image.fromarray(edges).resize((1024, 1024))  # SDXL res

# -----------------------------
# Generate with Variations
# -----------------------------
for i in range(args.num_images):
    # Dynamic Prompt with Weights
    base = random.choice(anunnaki_bases)
    addon = random.choice(quantum_futuristic)
    full_prompt = f"{base}{addon}, highly detailed, cinematic masterpiece"
    if args.prompt:
        full_prompt = args.prompt

    print(f"Generating {i+1}/{args.num_images}: {full_prompt}")
    print(f"Seed: {SEED + i}")

    generator = torch.Generator(device=DEVICE).manual_seed(SEED + i)

    if args.use_img2img and args.init_image:
        init_img = load_image(args.init_image).resize((1024, 1024))
        image = pipe(
            full_prompt,
            image=init_img,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator
        ).images[0]
    else:
        image = pipe(
            full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            controlnet_conditioning_image=control_image if args.use_controlnet else None
        ).images[0]

    # Refine
    refined = refiner(
        full_prompt,
        image=image,
        num_inference_steps=20,  # Quick refine
        guidance_scale=7.0,
        generator=generator
    ).images[0]

    # Optional Upscale (Simple 2x bicubic + sharpen)
    if args.upscale:
        refined = refined.resize((refined.width * 2, refined.height * 2), Image.BICUBIC)
        refined = TF.adjust_sharpness(refined, sharpness_factor=1.2)

    refined.save(f"anunnaki_quantum_{SEED + i}.png")
    print(f"Saved: anunnaki_quantum_{SEED + i}.png")
