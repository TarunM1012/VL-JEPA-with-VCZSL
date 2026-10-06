#!/bin/bash
# Feature extraction on Narval. Usage:
#   sbatch --export=ALL,BACKBONE=vjepa,EXTRA="--split test --limit 1000" scripts/extract_narval.sh   # timing run
#   sbatch --export=ALL,BACKBONE=vjepa scripts/extract_narval.sh                                     # full
#   sbatch --export=ALL,BACKBONE=clip  scripts/extract_narval.sh
#   sbatch --export=ALL,BACKBONE=vjepa,EXTRA="--split test --reverse" scripts/extract_narval.sh
#
# Create the log dir once before submitting:  mkdir -p logs
#SBATCH --job-name=extract
#SBATCH --account=def-fqureshi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load python/3.10 cuda/12.2
source ~/venvs/vczsl/bin/activate

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SSV2_VIDEO_ROOT=${SSV2_VIDEO_ROOT:-/scratch/tarunm10/datasets/ssv2/20bn-something-something-v2}
export STHCOM_SPLIT_ROOT=${STHCOM_SPLIT_ROOT:-/scratch/tarunm10/datasets/sth_com/data_split/generalized}
export FEATURE_ROOT=${FEATURE_ROOT:-/scratch/tarunm10/features/sth_com}

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
python extract_features.py --backbone "${BACKBONE:?set BACKBONE=vjepa or clip}" ${EXTRA:-}