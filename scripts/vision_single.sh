#!/bin/bash

#SBATCH --job-name=kp

#SBATCH --output=/home/jia/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=danjia1992@gmail.com

#SBATCH --partition=chimay

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G

#SBATCH --gres=gpu:1

#SBATCH --time=1-00:00:00

#SBATCH --signal=TERM@120

WS_DIR="$HOME/git/3d/3d_obj_det/drow-v2/v3/dr_spaam"
SCRIPT="bin/train_kp.py"

cd ${WS_DIR}

wandb on

srun --unbuffered python ${SCRIPT} --cfg cfgs/rpn_cfgs/ref_bl.yaml
