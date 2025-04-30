import torch
from diffusers import (
    StableDiffusionGLIGENPipeline,
    DPMSolverMultistepScheduler,
    ControlNetUnionModel,
    StableDiffusionXLControlNetUnionPipeline,
)
from PIL import Image
from pathlib import Path
import json

def load_gligen_pipe(
    model_id: str = "masterful/gligen-1-4-generation-text-box",
    variant: str = "fp16",
    device: str = "cuda",
) -> StableDiffusionGLIGENPipeline:
    """
    Load the GLIGEN generation-text-box pipeline from Hugging Face hub,
    with NSFW safety checker disabled to avoid false positives.
    """
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        model_id,
        variant=variant,
        torch_dtype=torch.float16,
    ).to(device)

    # disable the safety checker so outputs aren't blacked out
    pipe.safety_checker = None

    # swap in a DPMSolverMultistep scheduler if you prefer
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_controlnet_union_pipe(
    union_model_id: str = "xinsir/controlnet-union-sdxl-1.0",
    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "cuda",
    offload_cpu: bool = True,
) -> StableDiffusionXLControlNetUnionPipeline:
    """
    Load the ControlNet-Union pipeline from Hugging Face hub,
    safety checker disabled to prevent NSFW filtering.
    """
    controlnet = ControlNetUnionModel.from_pretrained(
        union_model_id,
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    if offload_cpu:
        pipe.enable_model_cpu_offload()
    return pipe


def run_gligen_generate(
    pipe: StableDiffusionGLIGENPipeline,
    prompt: str,
    output_path: Path,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 25,
    phrases: list[str] = None,
    locations: list[list[float]] = None,
) -> Image.Image:
    """
    Generate a full image with GLIGEN using text-box guidance.
    """
    if phrases is None:
        phrases = ["hair", "eyes", "scar"]
    if locations is None:
        locations = [
            [0.3, 0.1, 0.7, 0.3],
            [0.35, 0.4, 0.65, 0.5],
            [0.4, 0.6, 0.5, 0.7],
        ]

    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        gligen_boxes=locations,
        gligen_phrases=phrases,
    )
    img = output.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return img


def run_controlnet_union(
    pipe: StableDiffusionXLControlNetUnionPipeline,
    prompt: str,
    control_image: Image.Image,
    control_mode: int,
    output_path: Path,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 30,
) -> Image.Image:
    img_in = control_image.convert("RGB").resize((width, height))
    output = pipe(
        prompt=[prompt],
        control_image=[img_in],
        control_mode=[control_mode],
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
    )
    result = output.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return result


def main_interactive_loop():
    gligen_pipe = load_gligen_pipe()
    controlnet_pipe = load_controlnet_union_pipe()
    last_final = None
    base_output = Path("outputs/interactive")
    base_output.mkdir(parents=True, exist_ok=True)

    iteration = 0
    while True:
        prompt = input("Enter prompt (or 'exit'): ").strip()
        if prompt.lower() == 'exit':
            break
        print(f"Running iteration {iteration} for prompt: '{prompt}'")

        if last_final is not None:
            use_last = input("Reuse last final image as GLIGEN input? (y/n): ").strip().lower()
        else:
            use_last = 'n'

        if use_last == 'y' and last_final is not None:
            init_img = last_final
        else:
            path = input("Enter path to reference image (or Enter to use blank): ").strip()
            if path.lower() == 'exit':
                break
            if path:
                init_img = Image.open(path)
                use_full = input("Send full image to GLIGEN? (y/n): ").strip().lower()
                if use_full != 'y':
                    intermediate = init_img.resize((512,512))
                    gligen_out = None
            else:
                init_img = Image.new("RGB", (512,512), "white")
                gligen_out = None

        if ('use_full' in locals() and use_full == 'y') or (last_final is not None and use_last=='y') or (last_final is None and not path):

            with open("gligen_boxes.json", "r") as f:
                regions_list = json.load(f)
            phrases = [r["phrase"] for r in regions_list]
            boxes   = [r["box"]    for r in regions_list]
            gligen_out = base_output / f"iter_{iteration}_gligen.png"
            intermediate = run_gligen_generate(
            pipe=gligen_pipe,
            prompt=prompt,
            output_path=gligen_out,
            phrases = phrases,
            locations = boxes
            )
        print(f"Saved GLIGEN output to {gligen_out}")

        # Run ControlNet
        ctrl_out = base_output / f"iter_{iteration}_controlnet.png"
        final = run_controlnet_union(
            pipe=controlnet_pipe,
            prompt=prompt,
            control_image=intermediate,
            control_mode=6,
            output_path=ctrl_out
        )
        print(f"Saved ControlNet output to {ctrl_out}")

        last_final = final
        iteration += 1
        print("---")
    print("Interactive loop exited.")

if __name__ == "__main__":
    main_interactive_loop()
