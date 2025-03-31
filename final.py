"""
two_stage_with_plots.py

A script that:
  1) Takes user input describing a suspect's face.
  2) Uses GLIGEN for bounding-box-based generation (intermediate).
  3) Uses ControlNet to structurally refine that image (final).
  4) Plots both images separately with matplotlib.

Dependencies:
  pip install diffusers transformers controlnet_aux omegaconf matplotlib safetensors

Ensure you have:
  - gligen_inference.py in the same folder (or adjust import).
  - The correct paths for your GLIGEN and ControlNet++ checkpoints.

Run:
  python two_stage_with_plots.py
"""

import os
import argparse
import torch
from PIL import Image
from pathlib import Path
from argparse import Namespace
import matplotlib.pyplot as plt


# ------------------------------
# Import the GLIGEN "run()" function
# ------------------------------
# Make sure your gligen_inference.py is in the same directory or in PYTHONPATH
from gligen_inference import run as gligen_run

# ------------------------------
# Diffusers + ControlNet
# ------------------------------
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from controlnet_aux import LineartDetector


def stage1_generate_with_gligen(
    gligen_ckpt_path: str,
    prompt: str,
    output_path: str = "gligen_face.png"
):
    """
    Generates an image using GLIGEN with bounding-box constraints for the face.
    Returns the path to the saved output image.
    """
    # We'll use some example bounding boxes for "hair," "eyes," "mouth," etc.
    # Adjust as you like. Format is [x0, y0, x1, y1] in normalized coords.
    meta = {
        "ckpt": gligen_ckpt_path,
        "prompt": prompt,
        "phrases": ["hair", "eyes", "scar"],  # example feature labels
        "locations": [
            [0.3, 0.1, 0.7, 0.3],  # bounding box for hair
            [0.35, 0.4, 0.65, 0.5], # bounding box for eyes
            [0.4, 0.6, 0.5, 0.7],  # bounding box for scar
        ],
        "save_folder_name": "gligen_output"
    }

    # Build gligen_inference arguments
    gligen_args = Namespace(
        folder="generation_samples",
        batch_size=1,
        no_plms=False,
        guidance_scale=7.5,
        negative_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    )
    # Run GLIGEN
    print(gligen_args)
    gligen_run(meta, gligen_args, starting_noise=None)

    # By default, saves to generation_samples/gligen_output/0.png
    out_path = Path(gligen_args.folder) / meta["save_folder_name"] / "0.png"
    if not out_path.exists():
        raise FileNotFoundError(f"GLIGEN output not found at {out_path}")

    # Rename/copy to desired output filename
    Image.open(out_path).convert("RGB").save(output_path)
    print(f"[GLIGEN] Wrote intermediate image: {output_path}")

    return output_path


def stage2_refine_with_controlnet(
    controlnet_checkpoint: str,
    input_image_path: str,
    prompt: str,
    output_path: str = "final_refined_face.png",
    device: str = "cuda"
):
    """
    Uses ControlNet with a line-art detector to refine the structure of the
    GLIGEN-generated face. Saves and returns path to final image.
    """
    # Load lineart detector
    lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")

    # Load intermediate face
    intermediate_img = Image.open(input_image_path).convert("RGB").resize((512, 512))

    # Convert to line art
    control_image = lineart_detector(intermediate_img)
    control_image.save("control_image_lineart.png")
    print("[ControlNet] Generated line art -> control_image_lineart.png")

    # Load ControlNet checkpoint + pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # or whichever base SD you want
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    generator = torch.manual_seed(42)
    final_image = pipe(
        prompt=prompt,
        image=control_image,
        num_inference_steps=30,
        generator=generator
    ).images[0]

    final_image.save(output_path)
    print(f"[ControlNet] Wrote final image: {output_path}")

    return output_path


def main():
    # ------------------------------
    # Ask the user for text description
    # ------------------------------
    suspect_description = input(
        "Enter a description of the suspect's face (e.g., 'A male face with short brown hair, large eyes, a scar on left cheek'): "
    )
    if not suspect_description.strip():
        suspect_description = "A male face with short hair, large eyes, a scar on left cheek"

    # ------------------------------
    # Paths to your models
    # ------------------------------
    gligen_ckpt = "./gligen_checkpoints/checkpoint_generation_text.pth"
    controlnet_ckpt = "./ControlNet-checkpoints"  # or huggingface ID or local path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------
    # Stage 1: GLIGEN bounding-box generation
    # ------------------------------
    intermediate_path = "gligen_face.png"
    stage1_generate_with_gligen(
        gligen_ckpt_path=gligen_ckpt,
        prompt=suspect_description,
        output_path=intermediate_path
    )

    # ------------------------------
    # Stage 2: ControlNet refinement
    # ------------------------------
    final_path = "final_refined_face.png"
    stage2_refine_with_controlnet(
        controlnet_checkpoint=controlnet_ckpt,
        input_image_path=intermediate_path,
        prompt=suspect_description + ", photorealistic, high detail",
        output_path=final_path,
        device=device
    )

    # ------------------------------
    # Plot the intermediate and final images
    # ------------------------------
    # Important: 1) use matplotlib, 2) each chart is its own figure, 3) no custom colors/styles

    # Intermediate
    inter_img = Image.open(intermediate_path)
    plt.figure()
    plt.imshow(inter_img)
    plt.title("Intermediate (GLIGEN) Generation")
    plt.axis('off')
    plt.show()

    # Final
    final_img = Image.open(final_path)
    plt.figure()
    plt.imshow(final_img)
    plt.title("Final (ControlNet) Refined")
    plt.axis('off')
    plt.show()

    print("Done. Plotted intermediate and final images.")


if __name__ == "__main__":
    main()
