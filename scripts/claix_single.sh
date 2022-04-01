#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=train_nct

#SBATCH --output=/home/yx643192/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=danjia1992@gmail.com

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=3G

#SBATCH --gres=gpu:pascal:1

#SBATCH --time=1-00:00:00

#SBATCH --signal=TERM@120

#SBATCH --partition=c16g



source $HOME/.zshrc
conda activate torch10

WS_DIR="$HOME/git/dr-spaam-experimental/dr_spaam"
SCRIPT="bin/train.py"

cd ${WS_DIR}

wandb on

srun --unbuffered python ${SCRIPT} --cfg cfgs/drow5.yaml
