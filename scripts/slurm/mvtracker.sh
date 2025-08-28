#!/bin/bash
#SBATCH --job-name=mvtracker_200000_june2025
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --account=a136
#SBATCH --time=12:00:00
#SBATCH --dependency=singleton
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=frano.rajic@inf.ethz.ch
#SBATCH --output=./logs/slurm_logs/%x-%j.out
#SBATCH --error=./logs/slurm_logs/%x-%j.out
#SBATCH --signal=USR1@60

set -euo pipefail
set -x
cat $0
DIR=$(realpath .)

# Wrap the commands
CMD="
source /users/fraji/venvs/spa10/bin/activate
cd $DIR

python train.py model=mvtracker \
  trainer.num_steps=200000 \
  trainer.eval_freq=10000 \
  trainer.viz_freq=10000 \
  trainer.save_ckpt_freq=500 \
  trainer.lr=0.0005 \
  datasets.train.traj_per_sample=2048 \
  model.updatetransformer_type=cotracker2 \
  reproducibility.seed=36 \
  trainer.precision=bf16-mixed \
  modes.do_initial_static_pretrain=false \
  trainer.augment_train_iters=false \
  model.apply_sigmoid_to_vis=false \
  logging.log_wandb=true \
  experiment_path=logs/${SLURM_JOB_NAME}
"

# Execute within the container
srun -ul --environment=my_pytorch_env bash -c "$CMD"
