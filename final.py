import io
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# -------------------------------
# Import from your existing code
# (Make sure gligen_inference.py is accessible)
# -------------------------------
from gligen_inference import run as gligen_run
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from controlnet_aux import LineartDetector

###########################################################################
#      FIXED CHECKPOINT PATHS (unchanged), references your existing setup #
###########################################################################
GLIGEN_CKPT_TEXT_ONLY       = "./gligen_checkpoints/checkpoint_generation_text.pth"
GLIGEN_CKPT_TEXT_AND_IMAGE  = "./gligen_checkpoints/checkpoint_generation_text.pth"
CONTROLNET_CKPT_PATH        = "./ControlNet-checkpoints"

device = "cuda" if torch.cuda.is_available() else "cpu"

###########################################################################
#  GLOBAL DICT FOR IN-MEMORY IMAGE "PATHS"                                #
#  We'll monkey-patch Image.open so that, if someone tries to open a      #
#  "fake path" from this dict, we return the stored PIL image.           #
###########################################################################
_image_memory_storage = {}  # key: string, value: PIL.Image

_old_image_open = Image.open  # Original, unpatched Image.open

def _mock_image_open(path, *args, **kwargs):
    """
    If path is in _image_memory_storage, return that PIL image instead of
    reading from disk. Otherwise, call the original Image.open.
    """
    if path in _image_memory_storage:
        # Return a copy so we don't mutate the original
        return _image_memory_storage[path].copy()
    else:
        return _old_image_open(path, *args, **kwargs)

###########################################################################
#  1) HELPER: CAPTURE GLIGEN OUTPUT IN MEMORY INSTEAD OF DISK             #
###########################################################################
def _run_gligen_in_memory(meta, gligen_args):
    """
    Wraps gligen_inference.run() so it doesn't write to disk
    but instead returns the PIL image in memory.

    We also monkey-patch Image.open to intercept "fake paths" from meta["images"].
    """
    import io
    from PIL import Image

    # We'll monkey-patch *both* Image.save (to capture outputs)
    # AND Image.open (to load any in-memory "fake path" references).
    original_save = Image.Image.save

    buffer_dict = {"pil_img": None}

    def mock_save(self, fp, *args, **kwargs):
        # capture the final output in memory
        buf = io.BytesIO()
        original_save(self, buf, format='PNG')
        buf.seek(0)
        buffer_dict["pil_img"] = Image.open(buf).copy()
        buf.close()

    # Patch both:
    Image.open = _mock_image_open
    Image.Image.save = mock_save
    try:
        # Run the GLIGEN pipeline
        gligen_run(meta, gligen_args, starting_noise=None)
    finally:
        # Restore the original methods
        Image.open = _old_image_open
        Image.Image.save = original_save

    if buffer_dict["pil_img"] is None:
        raise RuntimeError("GLIGEN did not produce an image in memory.")

    return buffer_dict["pil_img"]

###########################################################################
#  2) GLIGEN bounding-box generation (hair, eyes, scar)                   #
###########################################################################
def generate_gligen_default_boxes(prompt: str) -> Image.Image:
    """
    Generates a face (PIL image, 512x512) with GLIGEN using default bounding boxes
    (hair, eyes, scar). Uses your text-only checkpoint.
    """
    from argparse import Namespace

    meta = {
        "ckpt": GLIGEN_CKPT_TEXT_ONLY,
        "prompt": prompt,
        "phrases": ["hair", "eyes", "scar"],
        "locations": [
            [0.3, 0.1, 0.7, 0.3],   # bounding box for hair
            [0.35, 0.4, 0.65, 0.5], # bounding box for eyes
            [0.4, 0.6, 0.5, 0.7],   # bounding box for scar
        ],
        "save_folder_name": "gligen_temp"  # not actually used
    }

    gligen_args = Namespace(
        folder="generation_samples",
        batch_size=1,
        no_plms=False,
        guidance_scale=7.5,
        negative_prompt=(
            "longbody, lowres, bad anatomy, bad hands, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality"
        ),
    )

    result = _run_gligen_in_memory(meta, gligen_args)
    return result.convert("RGB").resize((512, 512))

###########################################################################
#  3) GLIGEN with a full-image reference bounding box                     #
###########################################################################
def generate_gligen_full_image(prompt: str, ref_image_path: str) -> Image.Image:
    """
    Sends an entire reference image into GLIGEN as a bounding box covering
    the entire canvas. Requires a text+image checkpoint.
    """
    from argparse import Namespace

    meta = {
        "ckpt": GLIGEN_CKPT_TEXT_AND_IMAGE,
        "prompt": prompt,
        "images": [ref_image_path],               # single bounding box
        "phrases": ["placeholder"],               # matching length
        "locations": [[0.0, 0.0, 1.0, 1.0]],       # fill entire canvas
        "image_mask": [1],
        "text_mask": [1],
        "save_folder_name": "gligen_temp"
    }

    gligen_args = Namespace(
        folder="generation_samples",
        batch_size=1,
        no_plms=False,
        guidance_scale=7.5,
        negative_prompt=(
            "longbody, lowres, bad anatomy, bad hands, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality"
        ),
    )

    result = _run_gligen_in_memory(meta, gligen_args)
    return result.convert("RGB").resize((512, 512))

###########################################################################
#  4) GLIGEN with a full-image reference bounding box (from a PIL image)  #
###########################################################################
def generate_gligen_full_image_from_pil(prompt: str, pil_image: Image.Image) -> Image.Image:
    """
    Same as generate_gligen_full_image, but the "reference image" is a PIL object in memory.
    We assign it a 'fake path' in _image_memory_storage and pass that to GLIGEN.
    """
    from argparse import Namespace
    import uuid

    # 1) Create a unique "fake path"
    fake_path = f"inmemory://{uuid.uuid4()}"
    # 2) Store the PIL image in the global dictionary
    _image_memory_storage[fake_path] = pil_image.convert("RGB").resize((512, 512))

    # 3) Prepare meta
    meta = {
        "ckpt": GLIGEN_CKPT_TEXT_AND_IMAGE,
        "prompt": prompt,
        "images": [fake_path],       # we pass the fake path
        "phrases": ["placeholder"],
        "locations": [[0.0, 0.0, 1.0, 1.0]],
        "image_mask": [1],
        "text_mask": [1],
        "save_folder_name": "gligen_temp"
    }

    gligen_args = Namespace(
        folder="generation_samples",
        batch_size=1,
        no_plms=False,
        guidance_scale=7.5,
        negative_prompt=(
            "longbody, lowres, bad anatomy, bad hands, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality"
        ),
    )

    result = _run_gligen_in_memory(meta, gligen_args)
    return result.convert("RGB").resize((512, 512))

###########################################################################
#  5) ControlNet line-art refinement                                      #
###########################################################################
def refine_with_controlnet_lineart(input_image: Image.Image, prompt: str) -> Image.Image:
    """
    Takes a PIL image, extracts line-art via LineartDetector,
    and refines with ControlNet, returning a new PIL image.
    """
    lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
    working_img = input_image.convert("RGB").resize((512, 512))
    control_image = lineart_detector(working_img)

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_CKPT_PATH, torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
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

    return final_image

###########################################################################
#  MAIN ITERATIVE LOOP                                                    #
###########################################################################
def main_iterative_loop():
    """
    Demonstration loop where each iteration:
      - Asks for a text prompt
      - (Optional) uses the last final image or a new user-provided image
        in GLIGEN, or bypasses GLIGEN entirely for ControlNet
      - Displays intermediate + final results in memory (matplotlib)
      - Stores final image in last_final_image for the next iteration
    """
    import matplotlib.pyplot as plt
    last_final_image = None

    while True:
        print("\n--- Iterative Face Creation ---")
        user_prompt = input("Enter a description (or 'exit'): ")
        if user_prompt.strip().lower() == "exit":
            print("Exiting the loop.")
            break

        # Prompt user if they want to re-use the last final image in GLIGEN
        # or provide a new image path
        if last_final_image is not None:
            reuse = input("Use last final image as GLIGEN reference? (y/n): ").strip().lower()
        else:
            reuse = "n"

        if reuse == "y" and last_final_image is not None:
            # 1) Feed last_final_image to GLIGEN as a full bounding box (text+image checkpoint)
            print("[GLIGEN] Using last final image as reference bounding box...")
            try:
                intermediate_img = generate_gligen_full_image_from_pil(user_prompt, last_final_image)
                plt.figure()
                plt.imshow(intermediate_img)
                plt.title("Intermediate (GLIGEN from last_final_image)")
                plt.axis('off')
                plt.show()

                # 2) ControlNet refine
                print("[ControlNet] Refining structure via line-art...")
                final_img = refine_with_controlnet_lineart(intermediate_img, user_prompt)
                plt.figure()
                plt.imshow(final_img)
                plt.title("Final (ControlNet) Refined")
                plt.axis('off')
                plt.show()

                last_final_image = final_img

            except Exception as e:
                print(f"Error reusing last image in GLIGEN: {e}")
                continue

        else:
            # If not re-using last_final_image, see if we have a new user-provided path
            user_image_path = input("Optionally enter path to a reference image (press Enter to skip): ").strip()
            if user_image_path.lower() == "exit":
                print("Exiting the loop.")
                break

            if user_image_path:
                # The user gave an image path
                try:
                    test_img = Image.open(user_image_path).convert("RGB")
                except Exception as e:
                    print(f"Could not open user image: {e}")
                    continue

                # Ask user if they want to feed the entire image to GLIGEN
                choice = input("Send entire image to GLIGEN first? (y/n): ").strip().lower()
                if choice.startswith('y'):
                    # Stage 1: GLIGEN full bounding box
                    try:
                        intermediate_img = generate_gligen_full_image(user_prompt, user_image_path)
                        plt.figure()
                        plt.imshow(intermediate_img)
                        plt.title("Intermediate (GLIGEN: Full Image Box)")
                        plt.axis('off')
                        plt.show()

                        # Stage 2: ControlNet
                        final_img = refine_with_controlnet_lineart(intermediate_img, user_prompt)
                        plt.figure()
                        plt.imshow(final_img)
                        plt.title("Final (ControlNet) Refined from GLIGEN Output")
                        plt.axis('off')
                        plt.show()

                        last_final_image = final_img

                    except Exception as e:
                        print(f"Error in GLIGEN full-image approach: {e}")
                        continue

                else:
                    # Skip GLIGEN, just refine with ControlNet
                    final_img = refine_with_controlnet_lineart(test_img.resize((512,512)), user_prompt)
                    plt.figure()
                    plt.imshow(final_img)
                    plt.title("Final (ControlNet) [User Image Provided, skipped GLIGEN]")
                    plt.axis('off')
                    plt.show()

                    last_final_image = final_img

            else:
                # No user image => do default GLIGEN bounding boxes
                print("[GLIGEN] Generating from default bounding boxes (hair, eyes, scar)...")
                intermediate_img = generate_gligen_default_boxes(prompt=user_prompt)
                plt.figure()
                plt.imshow(intermediate_img)
                plt.title("Intermediate (GLIGEN: Default Boxes)")
                plt.axis('off')
                plt.show()

                print("[ControlNet] Refining structure via line-art...")
                final_img = refine_with_controlnet_lineart(intermediate_img, user_prompt)
                plt.figure()
                plt.imshow(final_img)
                plt.title("Final (ControlNet) Refined")
                plt.axis('off')
                plt.show()

                last_final_image = final_img

        # Loop again or exit
        cont = input("Press Enter to do another iteration, or type 'exit' to quit: ")
        if cont.strip().lower() == "exit":
            print("Exiting the loop.")
            break

###########################################################################
#  RUN if called directly                                                 #
###########################################################################
if __name__ == "__main__":
    main_iterative_loop()
