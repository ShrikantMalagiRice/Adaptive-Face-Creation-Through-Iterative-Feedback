import json
from pathlib import Path

import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from app import (load_gligen_generate_pipe,load_gligen_inpaint_pipe,load_controlnet_union_pipe,run_gligen_generate,run_gligen_inpaint,extract_region_boxes,run_controlnet_union,)

def calculate_clip_score(model, processor, image, text, device):
    inputs = processor(text=[text], images=[image],
                       return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_feats = model.get_image_features(pixel_values=inputs.pixel_values)
        txt_feats = model.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    return (img_feats * txt_feats).sum().item()

def batch_process_prompts(
    json_path: str,
    output_dir: str = "outputs/batch",
    csv_path: str    = "outputs/batch/results.csv",
    device: str      = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    gen_pipe     = load_gligen_generate_pipe(device=device)
    inpaint_pipe = load_gligen_inpaint_pipe(device=device)
    controlnet_pipe = load_controlnet_union_pipe()

    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    # remove any old CSV so header logic works
    if csv_file.exists():
        csv_file.unlink()

    with open("gligen_boxes.json", "r") as f:
            regions = json.load(f)
    general_phrases = [r["phrase"] for r in regions]
    general_boxes = [r["box"] for r in regions]

    # load prompts+feedback
    with open(json_path, "r") as f:
        entries = json.load(f)

    for i, entry in enumerate(entries):
        prompt   = entry["prompt"]
        feedback = entry["feedback"]

        init_path = out_dir / f"{i:03d}_initial.png"
        fb_path   = out_dir / f"{i:03d}_feedback.png"

        # 1) generate
        init_img = run_gligen_generate(pipe=gen_pipe,
                                       prompt=prompt,
                                       output_path=init_path,
                                       phrases=general_phrases,
                                       locations = general_boxes)
        boxes_dict = extract_region_boxes(init_img)
        feedback_phrases = list(boxes_dict.keys())
        feedback_boxes   = list(boxes_dict.values())
        init_cntrl_net_img = final = run_controlnet_union(
            control_mode=4,
            pipe=controlnet_pipe,
            prompt=prompt,
            control_image=init_img,
            output_path=init_path
        )
        # 2) inpaint with feedback
        fb_img = run_gligen_inpaint(pipe=inpaint_pipe,
                                    prompt= prompt + feedback,
                                    output_path=fb_path,
                                    init_image=init_img,
                                    phrases = feedback_phrases,
                                    locations = feedback_boxes)
        fb_img_cntrl_net_img = final = run_controlnet_union(
            control_mode=4,
            pipe=controlnet_pipe,
            prompt=prompt,
            control_image=fb_img,
            output_path=fb_path
        )
        # 3) clip scores
        score_init = calculate_clip_score(clip_model, clip_processor,
                                          init_cntrl_net_img, prompt, device)
        score_fb   = calculate_clip_score(clip_model, clip_processor,
                                          fb_img_cntrl_net_img,   prompt + feedback, device)

        # 4) build one‐row DataFrame
        row = {
            "Prompt": prompt,
            "output_file_initial_name": init_path.name,
            "CLIP-score-initial": round(score_init, 4),
            "feedback": feedback,
            "outputfile_name_feedback": fb_path.name,
            "clip_score_final": round(score_fb, 4),
        }
        df_row = pd.DataFrame([row])

        # 5) append to CSV
        df_row.to_csv(csv_file, mode="a",
                      header=not csv_file.exists(),
                      index=False)

        print(f"Saved results for {init_path.name} & {fb_path.name}")

if __name__ == "__main__":
    batch_process_prompts(
        json_path="prompts_feedback.json",
        output_dir="outputs/batch",
        csv_path="outputs/batch_results.csv"
    )
