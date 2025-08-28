#!/bin/bash
#SBATCH --job-name=repro-test-mvtracker
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --account=a-a03
#SBATCH --time=00:20:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

set -euo pipefail
set -x
cat $0
DIR=$(realpath .)
cd "$DIR"

# Use job ID for run directory
RUN1="logs/debug/test_repro_${SLURM_JOB_ID}_run1"
RUN2="logs/debug/test_repro_${SLURM_JOB_ID}_run2"
[[ ! -e "$RUN1" ]] || { echo "ERROR: $RUN1 already exists"; exit 1; }
[[ ! -e "$RUN2" ]] || { echo "ERROR: $RUN2 already exists"; exit 1; }

# Wrap the commands
CMD="
source /users/fraji/venvs/spa10/bin/activate
cd $DIR

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

# === Run 1 ===
python train.py +experiment=mvtracker_overfit \
  datasets.eval.names=[] \
  modes.tune_per_scene=false \
  trainer.num_steps=10 \
  reproducibility.deterministic=true \
  dataset.train.num_workers=4 \
  trainer.precision=32 \
  experiment_path=$RUN1

# === Run 2 ===
python train.py +experiment=mvtracker_overfit \
  datasets.eval.names=[] \
  modes.tune_per_scene=false \
  trainer.num_steps=10 \
  reproducibility.deterministic=true \
  dataset.train.num_workers=4 \
  trainer.precision=32 \
  experiment_path=$RUN2
"

# Execute within the container
srun -ul --environment=my_pytorch_env bash -c "$CMD"
