import torch
from diffusers import (
    StableDiffusionGLIGENPipeline,
    DPMSolverMultistepScheduler,
    ControlNetUnionModel,
    StableDiffusionXLControlNetUnionPipeline,
)
from PIL import Image, ImageDraw
from pathlib import Path
from controlnet_aux import LineartDetector
import json
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
REGION_LANDMARKS = {
    "left eye":        mp_face_mesh.FACEMESH_LEFT_EYE,
    "left eyebrow":    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    "lips":            mp_face_mesh.FACEMESH_LIPS,
    "nose":            mp_face_mesh.FACEMESH_NOSE,
    "right eye":       mp_face_mesh.FACEMESH_RIGHT_EYE,
    "right eyebrow":   mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
}

DUMMY_REGION_BOXES = {
    name: [0.0, 0.0, 1.0, 1.0]
    for name in REGION_LANDMARKS
}

mp_face_mesh = mp.solutions.face_mesh
_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,    
    min_detection_confidence=0.5,
)

def load_gligen_generate_pipe(
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
        torch_dtype=torch.float16,
        variant=variant,
        trust_remote_code=True,
    ).to(device)

    # disable the safety checker so outputs aren't blacked out
    pipe.safety_checker = None

    # swap in a DPMSolverMultistep scheduler if you prefer
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def load_gligen_inpaint_pipe(
    model_id: str = "masterful/gligen-1-4-inpainting-text-box",
    variant: str = "fp16",
    device: str = "cuda",
) -> StableDiffusionGLIGENPipeline:
    """
    Load the GLIGEN generation-text-box pipeline from Hugging Face hub,
    with NSFW safety checker disabled to avoid false positives.
    """
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant=variant,
        trust_remote_code=True,
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

def gligen_prompt_helper(prompt: str) -> str:
    prompt = "Image of a human face, human face nothing else, not animal human realistic faces with no gaps or deformities"+prompt

    return prompt

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

def extract_region_boxes(pil_img: Image.Image, padding: float = 0.03):
    """
    Runs MediaPipe on pil_img, returns dict: region_name -> [x0,y0,x1,y1],
    all coords normalized 0–1, with optional padding fraction.
    """
    # convert to RGB numpy
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    results = _face_mesh.process(img)
    if not results.multi_face_landmarks:
        return DUMMY_REGION_BOXES.copy()

    lm = results.multi_face_landmarks[0].landmark

    def _box_for(pairs):
        # flatten all indices from edge-pairs
        idxs = {i for edge in pairs for i in edge}
        xs = [lm[i].x for i in idxs]
        ys = [lm[i].y for i in idxs]
        x0, x1 = max(min(xs) - padding*(max(xs)-min(xs)), 0), min(max(xs) + padding*(max(xs)-min(xs)), 1)
        y0, y1 = max(min(ys) - padding*(max(ys)-min(ys)), 0), min(max(ys) + padding*(max(ys)-min(ys)), 1)
        return [x0, y0, x1, y1]

    boxes = {}
    for name, pairs in REGION_LANDMARKS.items():
        boxes[name] = _box_for(pairs)

    return boxes

def run_gligen_inpaint(
    pipe: StableDiffusionGLIGENPipeline,
    prompt: str,
    output_path: Path,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 25,
    init_image : Image.Image = None, 
    phrases: list[str] = None,
    locations: list[list[float]] = None,
    guidance_scale: float = 7.5,
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
    enhanced_prompt = gligen_prompt_helper(prompt)
    kwargs = {
        "prompt": prompt,
        "gligen_boxes": locations,
        "gligen_phrases": phrases,
        "height": height, "width": width,
        "num_inference_steps": num_inference_steps,
        "output_type": "pil",
        "gligen_inpaint_image": init_image.resize((width, height)),
    }    
    output = pipe(**kwargs)
    img = output.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return img

def  controlnet_enhance_prompt(prompt:str)->str:
    prompt = "You're a sketch artist sketching a human face, preserve original colors of face eyes hair eveything. face is main focus"+ prompt

    return prompt

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
    processor = LineartDetector.from_pretrained('lllyasviel/Annotators').to('cuda')
    controlnet_img = processor(img_in, output_type='pil')
    output = pipe(
        prompt=[prompt],
        control_image=[controlnet_img],
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
    gligen_generate_pipe = load_gligen_generate_pipe()
    gligen_inpaint_pipe = load_gligen_inpaint_pipe()
    controlnet_pipe = load_controlnet_union_pipe()
    last_final = None
    last_prompt = ""
    base_output = Path("outputs/interactive")
    base_output.mkdir(parents=True, exist_ok=True)

    iteration = 0
    while True:
        prompt = input("Enter prompt (or 'exit'): ").strip()
        if prompt.lower() == 'exit':
            break
        prompt = last_prompt + " " + prompt
        print(f"Running iteration {iteration} for prompt: '{prompt}'")

        if last_final is not None:
            use_last = input("Reuse last final image as GLIGEN input? (y/n): ").strip().lower()
        else:
            use_last = 'n'

        if use_last == 'y' and last_final is not None:
            init_img = last_final
            use_full = 'y'
        else:
            path = input("Enter path to reference image (or Enter to use blank): ").strip()
            if path.lower() == 'exit':
                break
            if path:
                init_img = Image.open(path)
                use_full = input("Send full image to GLIGEN? (y/n): ").strip().lower()
            else:
                init_img = Image.new("RGB", (512,512), "white")
                gligen_out = None
                use_full = 'n'
            
        with open("gligen_boxes.json", "r") as f:
            regions = json.load(f)
        phrases = [r["phrase"] for r in regions]
        boxes = [r["box"] for r in regions]

        gligen_out = base_output / f"iter_{iteration}_gligen.png"
        if use_full == 'y':
            # inpainting (feedback or reference image)

            boxes_dict = extract_region_boxes(init_img)
            phrases = list(boxes_dict.keys())
            boxes   = list(boxes_dict.values())

            intermediate = run_gligen_inpaint(
                pipe      = gligen_inpaint_pipe,
                prompt    = prompt,
                output_path=gligen_out,
                init_image = init_img,
                phrases    = phrases,
                locations  = boxes,
            )
        else:
            # pure generation
            intermediate = run_gligen_generate(
                pipe       = gligen_generate_pipe,
                prompt     = prompt,
                output_path= gligen_out,
                phrases    = phrases,
                locations  = boxes,
            )
        print(f"Saved GLIGEN output to {gligen_out}")

        # Run ControlNet
        ctrl_out = base_output / f"iter_{iteration}_controlnet.png"
        final = run_controlnet_union(
            control_mode=4,
            pipe=controlnet_pipe,
            prompt=prompt,
            control_image=intermediate,
            output_path=ctrl_out
        )
        print(f"Saved ControlNet output to {ctrl_out}")

        last_final = intermediate
        last_prompt = prompt
        iteration += 1
        print("---")
    print("Interactive loop exited.")

if __name__ == "__main__":
    main_interactive_loop()
