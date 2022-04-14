import torch
import torchvision
from torch import optim

import numpy as np
import argparse
import yaml
import os

from src.dataset.arcface_get_dataloader import get_train_dataloader, get_test_dataloader
from src.pipeline.pipeline import Pipeline
from src.model.get_model import get_model

def run_training(model, pipeline, cfg):
    # main train loop
    train_loader = get_train_dataloader(cfg)

    status = pipeline.train(model, train_loader)


def run_evaluation(model, pipeline, cfg):
    test_loader = get_test_dataloader(
        data_path=cfg["dataset"]["data_dir"], batch_size=cfg["dataloader"]["batch_size"], num_workers=cfg["dataloader"]["num_workers"]
    )
    pipeline.hw_evaluate(model, test_loader, tb_prefix="TEST")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--evaluation", default=False, action="store_true")
    parser.add_argument("--cfg", type=str, required=False, default="./cfgs/learn_arcface.yaml")
    parser.add_argument("--ckpt", type=str, required=False, default=None)
    parser.add_argument("--cont", default=False, action="store_true")
    args = parser.parse_args()

    # open config files
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["pipeline"]["Logger"]["backup_list"].append(args.cfg)

    model = get_model(cfg)
    model.cuda()

    pipeline = Pipeline(model, cfg["pipeline"])

    if args.ckpt:
        pipeline.load_ckpt(model, args.ckpt)
    elif args.cont and pipeline.sigterm_ckpt_exists():
        pipeline.load_sigterm_ckpt(model)

    # training or evaluation
    if not args.evaluation:
        run_training(model, pipeline, cfg)
    else:
        run_evaluation(model, pipeline, cfg)

    pipeline.close()
