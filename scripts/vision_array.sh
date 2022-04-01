#!/bin/bash

#SBATCH --job-name=train_PSG_small

#SBATCH --output=/home/jia/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=danjia1992@gmail.com

#SBATCH --partition=lopri

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G

#SBATCH --gres=gpu:1

#SBATCH --time=1-00:00:00

#SBATCH --signal=TERM@120

#SBATCH --array=1-5

WS_DIR="$HOME/git/3d/3d_obj_det/drow-v2/v3"
SCRIPT="train.py"

cd ${WS_DIR}

wandb on

file=`ls PSG_cfgs/PSG_small_*.yaml | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`

srun --unbuffered python ${SCRIPT} --cfg $file
