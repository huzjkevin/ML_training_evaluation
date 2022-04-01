#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=new_p

#SBATCH --output=/home/yx643192/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=danjia1992@gmail.com

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=3G

#SBATCH --gres=gpu:1

#SBATCH --time=2-00:00:00

#SBATCH --signal=TERM@120

#SBATCH --partition=c18g

#SBATCH --array=1-4

#SBATCH --account=rwth0485


source $HOME/.zshrc
conda activate torch10

WS_DIR="$HOME/git/dr-spaam-experimental/dr_spaam"
SCRIPT="bin/train.py"

cd ${WS_DIR}

wandb on

file=`ls cfgs/dr_spaam_*_dp.yaml | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`

srun --unbuffered python ${SCRIPT} --cfg $file
