# A hello world demonstration for the ML pipeline
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

    loss = loss_fn(logits, labels)
    rtn_dict["loss"] = loss
    tb_dict["loss"] = loss

    return loss, tb_dict, rtn_dict

def _model_eval_fn(model, eval_loader):
    return None
