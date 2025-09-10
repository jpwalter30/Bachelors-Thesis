# Image Feature Extraction (DINOv3) + Mini TabPFN Demo

This project extracts global image features using **DINOv3 ViT-B/16** (`facebook/dinov3-vitb16-pretrain-lvd1689m`).  
Optionally, a tiny **TabPFN demo** can be run (2 images per class, 2-shot train/test).

---

## Quick Start

1. **Install**
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

2. **Data layout**
   data/
  real/   # real images
  gen/    # generated images

3. **Freature extraction**
# Feature extraction
python extract_dinov3_features.py \
  --images_dir ./data \
  --hf_model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --out_dir ./outputs_dinov3

4. **Optional: Mini demo**

python extract_dinov3_features.py \
  --images_dir ./data \
  --hf_model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --out_dir ./outputs_dinov3 \
  --mode demo2


  