#!/bin/bash
#SBATCH --job-name=spatracker_multiview_128
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --account=a-a136-1
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

python train.py model=spatracker_multiview model.triplane_xres=128 model.triplane_yres=128 model.triplane_zres=128 \
  trainer.num_steps=200000 \
  trainer.eval_freq=10000 \
  trainer.viz_freq=10000 \
  trainer.save_ckpt_freq=500 \
  trainer.lr=0.001 \
  datasets.train.traj_per_sample=768 \
  reproducibility.seed=36 \
  trainer.precision=bf16-mixed \
  modes.do_initial_static_pretrain=true \
  trainer.augment_train_iters=true \
  experiment_path=logs/${SLURM_JOB_NAME}
"

# Execute within the container
srun -ul --environment=my_pytorch_env bash -c "$CMD"
