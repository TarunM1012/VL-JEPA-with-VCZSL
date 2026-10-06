#!/bin/bash
# Probes on cached features. Run AFTER extraction finishes. Usage:
#   sbatch --export=ALL,VERB=vjepa,OBJ=vjepa scripts/probe_narval.sh
#   sbatch --export=ALL,VERB=clip,OBJ=clip   scripts/probe_narval.sh
#   sbatch --export=ALL,VERB=vjepa,OBJ=clip  scripts/probe_narval.sh   # hybrid, after the two above
#SBATCH --job-name=probe
#SBATCH --account=def-fqureshi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load python/3.10 cuda/12.2
source ~/venvs/vczsl/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export STHCOM_SPLIT_ROOT=${STHCOM_SPLIT_ROOT:-/scratch/tarunm10/datasets/sth_com/data_split/generalized}
export FEATURE_ROOT=${FEATURE_ROOT:-/scratch/tarunm10/features/sth_com}
export PROBE_ROOT=${PROBE_ROOT:-/scratch/tarunm10/probe_runs}

cd "$SLURM_SUBMIT_DIR"
python probe.py --verb "${VERB:?}" --obj "${OBJ:?}" ${EXTRA:-}