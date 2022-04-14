import torch


import torch
import torch.nn as nn
import math


loss_fn = nn.CrossEntropyLoss()

def _model_fn(model, batch):
    rtn_dict, tb_dict = {}, {}
    #unpack data
    images, labels = batch

    #Move data to GPU
    images = images.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    emb, logits = model(images, labels)

    if torch.any(torch.isnan(logits)):
        print(1)
    if torch.any(torch.isnan(labels)):
        print(1)
    if torch.any(torch.isnan(images)):
        print(1)

    loss = loss_fn(logits, labels)

    if torch.isnan(loss):
        print(1)

    rtn_dict["loss"] = loss
    tb_dict["loss"] = loss

    return loss, tb_dict, rtn_dict

def _model_eval_fn(model, eval_loader):
    return None
