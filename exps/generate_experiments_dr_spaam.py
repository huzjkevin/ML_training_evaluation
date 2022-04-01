import os
import shutil
import yaml

_OUTPUT_DIR = "./exps/dr_spaam_exps"


def _generate_sbatch_file(fname, sbatch_args, cfg_fname, cmd_args=""):
    cfg_fname = os.path.abspath(cfg_fname)

    lines = [
        "source $HOME/.zshrc\n",
        "conda activate torch10\n",
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
Experiments relate to the DR-SPAAM paper
https://arxiv.org/abs/2004.14079
"""


def exp_ablation():
    """Ablation study DR-SPA and DR-AM"""

    cfg_dir = os.path.join(_OUTPUT_DIR, "exp_ablation")

    if os.path.exists(cfg_dir):
        shutil.rmtree(cfg_dir)

    os.makedirs(cfg_dir, exist_ok=False)

    exp_names = ("dr_spa", "dr_am")

    for fname in exp_names:
        with open("./cfgs/dr_spaam.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            cfg["dataset"]["DataHandle"]["data_dir"] = "./data/DROWv2-data"

        cfg["pipeline"]["Logger"]["tag"] = f"ablation_{fname}"

        yaml_fname = os.path.join(cfg_dir, f"{fname}.yaml")
        with open(yaml_fname, "w") as f:
            yaml.dump(cfg, f)

        # generate sbatch script
        sbatch_args = {
            "job-name": "exp_ablation",
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


def exp_main():
    """Compare three networks, DROW (T=1), DROW (T=5), and DR-SPAAM."""

    cfg_dir = os.path.join(_OUTPUT_DIR, "exp_main")

    if os.path.exists(cfg_dir):
        shutil.rmtree(cfg_dir)

    os.makedirs(cfg_dir, exist_ok=False)

    cfg_fnames = (
        "./cfgs/dr_spaam.yaml",
        "./cfgs/drow.yaml",
        "./cfgs/drow5.yaml",
    )
    for cfg_fname in cfg_fnames:
        with open(cfg_fname, "r") as f:
            cfg = yaml.safe_load(f)
            cfg["dataset"]["DataHandle"]["data_dir"] = "./data/DROWv2-data"

        fname = os.path.basename(cfg_fname).split(".")[0]

        cfg["pipeline"]["Logger"]["tag"] = f"main_{fname}"

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


def exp_spaam():
    """Hyper parameter selection for DR-SPAAM."""

    with open("./cfgs/dr_spaam.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        cfg["dataset"]["DataHandle"]["data_dir"] = "./data/DROWv2-data"

    # These values use the convention in the paper for depth (second entry).
    # The value should be divided by 2 to generate the config.
    width_alpha_combinations = [
        (7, 0.3),
        (7, 0.5),
        (7, 0.8),
        (11, 0.3),
        (11, 0.5),
        (11, 0.8),
        (15, 0.3),
        (15, 0.5),
        (15, 0.8),
    ]

    cfg_dir = os.path.join(_OUTPUT_DIR, "exp_spaam")

    if os.path.exists(cfg_dir):
        shutil.rmtree(cfg_dir)

    os.makedirs(cfg_dir, exist_ok=False)

    for idx, (w, alp) in enumerate(width_alpha_combinations):
        cfg["model"]["kwargs"]["window_size"] = w
        cfg["model"]["kwargs"]["alpha"] = alp

        fname = f"spaam_w{w}alp{alp}"
        cfg["pipeline"]["Logger"]["tag"] = fname

        yaml_fname = os.path.join(cfg_dir, f"{fname}.yaml")
        with open(yaml_fname, "w") as f:
            yaml.dump(cfg, f)

        # generate sbatch script
        sbatch_args = {
            "job-name": "exp_spaam",
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


def exp_cutout():
    """Hyper parameter selection for cutout."""

    with open("./cfgs/drow.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        cfg["dataset"]["DataHandle"]["data_dir"] = "./data/DROWv2-data"

    # These values use the convention in the paper for depth (second entry).
    # The value should be divided by 2 to generate the config.
    width_depth_pts_combinations = [
        (1.66, 2.0, 48),
        (1.66, 1.0, 48),
        (1.0, 2.0, 48),
        (1.0, 1.0, 48),
        (1.0, 1.0, 32),
        (1.0, 1.0, 40),
        (1.0, 1.0, 48),
        (1.0, 1.0, 56),
        (1.0, 1.0, 64),
    ]

    cfg_dir = os.path.join(_OUTPUT_DIR, "exp_cutout")

    if os.path.exists(cfg_dir):
        shutil.rmtree(cfg_dir)

    os.makedirs(cfg_dir, exist_ok=False)

    for idx, (w, d, p) in enumerate(width_depth_pts_combinations):
        cfg["dataset"]["cutout_kwargs"]["window_width"] = w
        cfg["dataset"]["cutout_kwargs"]["window_depth"] = d / 2
        cfg["dataset"]["cutout_kwargs"]["num_cutout_pts"] = p

        fname = f"cutout_w{w}d{d}p{p}"
        cfg["pipeline"]["Logger"]["tag"] = fname

        yaml_fname = os.path.join(cfg_dir, f"{fname}.yaml")
        with open(yaml_fname, "w") as f:
            yaml.dump(cfg, f)

        # generate sbatch script
        sbatch_args = {
            "job-name": "exp_cutout",
            "output": f"/home/yx643192/slurm_logs/%x_%J_{fname}.log",
            "mail-type": "ALL",
            "mail-user": "danjia1992@gmail.com",
            "cpus-per-task": "8",
            "mem-per-cpu": "3G",
            "gres": "gpu:1",
            "time": "1-00:00:00",
            "signal": "TERM@120",
            "partition": "c18g",
            "account": "rwth0485",
        }

        sh_fname = os.path.join(cfg_dir, f"{fname}.sh")
        _generate_sbatch_file(sh_fname, sbatch_args, yaml_fname)


def exp_stride():
    """Experiments related to temporal stride."""

    with open("./cfgs/dr_spaam.yaml", "r") as f:
        spaam_cfg = yaml.safe_load(f)
        spaam_cfg["dataset"]["DataHandle"]["data_dir"] = "./data/DROWv2-data"

    with open("./cfgs/drow5.yaml", "r") as f:
        drow5_cfg = yaml.safe_load(f)
        drow5_cfg["dataset"]["DataHandle"]["data_dir"] = "./data/DROWv2-data"

    stride_list = [1, 2, 3, 4, 5]

    cfg_dir = os.path.join(_OUTPUT_DIR, "exp_stride")

    if os.path.exists(cfg_dir):
        shutil.rmtree(cfg_dir)

    os.makedirs(cfg_dir, exist_ok=False)

    for idx, stride in enumerate(stride_list):
        spaam_cfg["dataset"]["DataHandle"]["scan_stride"] = stride
        drow5_cfg["dataset"]["DataHandle"]["scan_stride"] = stride

        spaam_fname = f"stride_spaam_s{stride}"
        drow5_fname = f"stride_drow5_s{stride}"
        spaam_cfg["pipeline"]["Logger"]["tag"] = spaam_fname
        drow5_cfg["pipeline"]["Logger"]["tag"] = drow5_fname

        spaam_yaml_fname = os.path.join(cfg_dir, f"{spaam_fname}.yaml")
        drow5_yaml_fname = os.path.join(cfg_dir, f"{drow5_fname}.yaml")

        with open(spaam_yaml_fname, "w") as f:
            yaml.dump(spaam_cfg, f)

        with open(drow5_yaml_fname, "w") as f:
            yaml.dump(drow5_cfg, f)

        # generate sbatch script
        sbatch_args = {
            "job-name": "exp_stride",
            "output": f"/home/yx643192/slurm_logs/%x_%J_{spaam_fname}.log",
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

        spaam_ckpt_file = "./ckpts/dr_spaam_e40.pth"
        spaam_cmd_args = f"--ckpt {spaam_ckpt_file} --evaluation"
        spaam_sh_fname = os.path.join(cfg_dir, f"{spaam_fname}.sh")
        _generate_sbatch_file(
            spaam_sh_fname, sbatch_args, spaam_yaml_fname, spaam_cmd_args
        )

        drow5_ckpt_file = "./ckpts/drow5_e40.pth"
        drow5_cmd_args = f"--ckpt {drow5_ckpt_file} --evaluation"
        sbatch_args["output"] = f"/home/yx643192/slurm_logs/%x_%J_{drow5_fname}.log"
        drow5_sh_fname = os.path.join(cfg_dir, f"{drow5_fname}.sh")
        _generate_sbatch_file(
            drow5_sh_fname, sbatch_args, drow5_yaml_fname, drow5_cmd_args
        )


if __name__ == "__main__":
    exp_cutout()
    exp_spaam()
    exp_stride()
    exp_main()
    exp_ablation()
