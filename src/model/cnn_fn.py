# A hello world demonstration for the ML pipeline
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

def _model_fn(model, data):
    rtn_dict, tb_dict = {}, {}
    #unpack data
    input, target = data

    #Move data to GPU
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    pred = model(input)

    loss = loss_fn(pred, target)
    rtn_dict["loss"] = loss
    tb_dict["loss"] = loss

    return loss, tb_dict, rtn_dict

def _model_eval_fn(model, eval_loader):
    return None
