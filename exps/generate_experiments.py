import os
import shutil
import yaml

_OUTPUT_DIR = "./exps/detr_exps"


def _generate_sbatch_file(fname, sbatch_args, cfg_fname, cmd_args=""):
    cfg_fname = os.path.abspath(cfg_fname)

    lines = [
        "source $HOME/.zshrc\n",
        "conda activate dr-spaam\n",
        "cd $HOME/git/dr-spaam-experimental/dr_spaam\n",
        "wandb on\n",
        f"srun --unbuffered python bin/train.py --cfg {cfg_fname} {cmd_args}\n",
    ]

    with open(fname, "w") as f:
        f.write("#!/usr/local_rwth/bin/zsh\n")

        for key, val in sbatch_args.items():
            f.write(f"#SBATCH --{key}={val}\n")

        for li in lines:
            f.write(li)


"""
Experiments relate to DeTr
"""


def exp_main():
    """Compare three networks, DROW (T=1) and DR-SPAAM."""

    cfg_dir = os.path.join(_OUTPUT_DIR, "exp_main")

    if os.path.exists(cfg_dir):
        shutil.rmtree(cfg_dir)

    os.makedirs(cfg_dir, exist_ok=False)

    cfg_fnames = ("./cfgs/detr.yaml", "./cfgs/drow.yaml")
    for cfg_fname in cfg_fnames:
        with open(cfg_fname, "r") as f:
            cfg = yaml.safe_load(f)
            cfg["dataset"]["DataHandle"]["data_dir"] = "./data/JRDB"
            cfg["dataset"]["DataHandle"]["tracking"] = "detr" in cfg_fname

        fname = os.path.basename(cfg_fname).split(".")[0]

        cfg["pipeline"]["Logger"]["tag"] = f"jrdb_{fname}"

        yaml_fname = os.path.join(cfg_dir, f"{fname}.yaml")
        with open(yaml_fname, "w") as f:
            yaml.dump(cfg, f)

        # generate sbatch script
        sbatch_args = {
            "job-name": "exp_main",
            "output": f"/home/yx643192/slurm_logs/%x_%J_{fname}.log",
            "mail-type": "ALL",
            "mail-user": "danjia1992@gmail.com",
            "cpus-per-task": "8",
            "mem-per-cpu": "3G",
            "gres": "gpu:1",
            "time": "2-00:00:00",
            "signal": "TERM@120",
            "partition": "c18g",
            "account": "rwth0485",
        }

        sh_fname = os.path.join(cfg_dir, f"{fname}.sh")
        _generate_sbatch_file(sh_fname, sbatch_args, yaml_fname)


if __name__ == "__main__":
    exp_main()
