#!/bin/bash
#SBATCH --job-name=dinov3_demo2
#SBATCH --partition=dev_gpu_il        # kleine Debug-Partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00               # kurz halten für schnelle Tests
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
echo "[$(date)] Node: $(hostname)  JobID: ${SLURM_JOB_ID}"

# Conda Env aktivieren
source /pfs/work9/workspace/scratch/ma_jwaltea-bachelor_thesis_tabpfn/miniconda3/bin/activate tabpfn_demo

# Caches ins Scratch legen
export HF_HOME="${PWD}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${PWD}/.cache/torch"
export PYTHONUNBUFFERED=1

mkdir -p outputs_dinov3

python extract_dinov3_features.py \
  --images_dir ./data \
  --hf_model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --out_dir ./outputs_dinov3 \
  --mode demo2

echo "[$(date)] Done."
echo "Outputs:"
ls -lh outputs_dinov3 || true
