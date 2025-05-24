# pose_generator_reference.py
"""
Reference script: SDXL + ControlNet(OpenPose) + IP‑Adapter FaceID
-----------------------------------------------------------------
Replicates the core of WeShopAI‑Pose‑Generator with fully open‑source
components using 🤗 Diffusers‑0.33+. The pipeline:

 1. Load SDXL‑base (text‑to‑img / img2img backbone).
 2. Attach an OpenPose ControlNet to steer the target pose.
 3. Attach an IP‑Adapter FaceID‑Plus LoRA to lock identity from a
    reference portrait.
 4. Detect the target pose map with controlnet_aux.OpenposeDetector.
 5. Run img2img conditioned on pose map + identity to produce a new
    image preserving clothes / background.

Requires a CUDA GPU (24 GB VRAM recommended) and Python ≥ 3.9.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    DDIMScheduler,
)
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection
from controlnet_aux import OpenposeDetector
from PIL import Image

# -----------------------------------------------------------------------------
# Model repo IDs
# -----------------------------------------------------------------------------
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"  # ([huggingface.co](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0?utm_source=chatgpt.com))
CONTROLNET_OPENPOSE = "thibaud/controlnet-openpose-sdxl-1.0"  # ([huggingface.co](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0?utm_source=chatgpt.com))
IP_ADAPTER_REPO = "h94/IP-Adapter-FaceID"  # ([huggingface.co](https://huggingface.co/h94/IP-Adapter-FaceID?utm_source=chatgpt.com))
IP_ADAPTER_WEIGHT = "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"  # weight inside repo


def parse_args():
    p = argparse.ArgumentParser(description="Pose‑transfer with SDXL + ControlNet + IP‑Adapter")
    p.add_argument("--reference", type=Path, required=False, help="Reference portrait (identity)", default="")
    p.add_argument("--target", type=Path, required=False, help="Target pose image (photo) OR pre‑made pose map", default="")
    p.add_argument("--prompt", type=str, default="best quality, high‑resolution photo", help="Positive prompt")
    p.add_argument("--negative", type=str, default="low quality, deformed, bad anatomy", help="Negative prompt")
    p.add_argument(
        "--steps", type=int, default=30, help="Number of diffusion inference steps (20–50 is typical)"
    )
    p.add_argument("--strength", type=float, default=0.8, help="Denoising strength for img2img")
    p.add_argument("--ip_scale", type=float, default=0.9, help="IP‑Adapter influence [0‑1]")
    p.add_argument("--out", type=Path, default=Path("output.png"), help="Where to save the generated image")
    return p.parse_args()


def main():
    args = parse_args()
    args.reference = r"C:\Users\Lab\Downloads\img_side.png"
    args.target = r"C:\Users\Lab\Downloads\output_split_imgs\Shay0_rp0_part1.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # ------------------------- Load backbone + ControlNet -------------------------
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_OPENPOSE, torch_dtype=dtype)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_BASE,
        controlnet=controlnet,
        torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    # ------------------------- Attach IP‑Adapter FaceID ---------------------------
    # FaceID variant expects its own image encoder.
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IP_ADAPTER_REPO, subfolder="models/image_encoder", torch_dtype=dtype
    )
    pipe.image_encoder = image_encoder  # replace encoder inside pipeline

    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder="sdxl_models",
        weight_name=IP_ADAPTER_WEIGHT,
    )
    pipe.set_ip_adapter_scale(args.ip_scale)

    # ------------------------- Prepare inputs ------------------------------------
    ref_image = load_image(str(args.reference))

    target_image = load_image(str(args.target))
    # If the target file is already a black‑on‑white pose map (single channel),
    # skip OpenPose detection.
    if target_image.mode != "RGB" or target_image.getextrema()[0][1] < 255:
        pose_map = target_image
    else:
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        pose_map: Image.Image = openpose(target_image)

    # ------------------------- Inference -----------------------------------------
    generated = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        image=target_image,  # initial latent for img2img (keeps clothing/background)
        controlnet_conditioning_image=pose_map,
        ip_adapter_image=ref_image,
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=6.5,
    ).images[0]

    generated.save(args.out)
    print(f"Saved → {args.out.absolute()}")


if __name__ == "__main__":
    main()
