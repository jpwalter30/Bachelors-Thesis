mkdir -p slurm outputs_dinov3 .cache/huggingface
cat > slurm/demo_a100.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=dinov3_demo2
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

# ---------- Setup ----------
set -euo pipefail
echo "[$(date)] Node: $(hostname)  JobID: ${SLURM_JOB_ID}"

# Conda aktivieren
source /pfs/work9/workspace/scratch/ma_jwaltea-bachelor_thesis_tabpfn/miniconda3/bin/activate tabpfn_demo

# Caches ins Scratch legen (schont $HOME-Quota)
export HF_HOME="${PWD}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${PWD}/.cache/torch"
export PYTHONUNBUFFERED=1

# (Optional) Erstlauf: HF-Login/Token nötig, falls das Modell auth erfordert:
# huggingface-cli login --token <HF_TOKEN>

# Ordner sicherstellen
mkdir -p outputs_dinov3

# ---------- Run demo ----------
python extract_dinov3_features.py \
  --images_dir ./data \
  --hf_model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --out_dir ./outputs_dinov3 \
  --mode demo2

echo "[$(date)] Done."
echo "Outputs:"
ls -lh outputs_dinov3 || true
EOF

# ausführbar machen
chmod +x slurm/demo_a100.sh
