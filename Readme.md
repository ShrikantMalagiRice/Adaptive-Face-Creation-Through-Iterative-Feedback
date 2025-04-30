# 🧠 GLIGEN + ControlNet Union Interactive Generation

This repository provides an interactive pipeline to generate high-quality images using [GLIGEN](https://huggingface.co/masterful/gligen-1-4-generation-text-box) and [ControlNet-Union](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0). The pipeline allows controlled generation via text prompts and reference image regions.

---

## 📁 Repository Contents

- `app.py`: Main entry point with an interactive CLI loop to generate images.
- `gligen_boxes.json`: Predefined bounding box regions + phrases for GLIGEN.
- `reservation.sh`: Script to reserve a GPU on `nots.rice.edu` (SLURM).
- `environment.sh`: Loads necessary modules and activates the Conda environment.
- `requirements.txt`: Python dependencies (Diffusers, HuggingFace, etc.).

---

## ⚙️ Prerequisites

- Access to [Rice University NOTS cluster](https://doi.org/10.36887/rice.nots)
- SLURM account with GPU access
- Mamba + Conda environment setup with PyTorch + CUDA 12.4

---

## 🚀 Quick Start (on `nots.rice.edu`)

1. **Login to NOTS**
   ```bash
   ssh <your_netid>@nots.rice.edu
2. **Reserve a GPU**
   ```bash
   source reservation.sh
3. **Load environment**
    ```bash
    source environment.sh
4. **Install dependencies (Only once, or if the environment is not preconfigured)**
    ```bash
    pip install -r requirements.txt
5. **Run the interactive app**
    ```bash
    python app.py