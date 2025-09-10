#!/usr/bin/env bash
#SBATCH --job-name=dinov3_demo2
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=00:15:00  
#SBATCH --output=slurm/Job_%A_%a.out

set -euo pipefail
echo "[$(date)] Node: $(hostname)  JobID: ${SLURM_JOB_ID}"

# Conda Env aktivieren
source /pfs/work9/workspace/scratch/ma_jwaltea-bachelor_thesis_tabpfn/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn_demo

# Caches ins Scratch legen
export HF_HOME="${PWD}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${PWD}/.cache/torch"
export PYTHONUNBUFFERED=1

mkdir -p outputs_dinov3

which python
python -V
python -c "import sys; print('PY:', sys.executable); import numpy, torch; print('numpy:', numpy.__version__); print('torch:', torch.__version__)"


# Skript starten
python extract_dinov3_features.py \
  --images_dir ./data \
  --hf_model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --out_dir ./outputs_dinov3 \
  --mode demo2

echo "[$(date)] Done."
echo "Outputs:"
ls -lh outputs_dinov3 || true
