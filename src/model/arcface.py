# from https://github.com/deepinsight/insightface/blob/eca1d9a6cd25653067e7293e27b8a2d0d2cd5415/recognition/arcface_torch/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math 

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5, num_classes=93431):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False
        # self.fc = nn.Linear(512, num_classes)
        self.weight = Parameter(torch.FloatTensor(num_classes, 512))
        nn.init.xavier_uniform_(self.weight)

        # nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, emb, labels):
        # logits = F.normalize(self.fc(emb))
        logits = F.linear(F.normalize(emb), F.normalize(self.weight)) # this has been tested the best so far but note that the ourput needs to be clamped
        # logits = F.normalize(F.linear(emb, self.weight))

        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2).clamp(0, 1))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)

        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        
        return logits

    @staticmethod
    def model_fn(model, batch_data):
        return _model_fn(model, batch_data)

    @staticmethod
    def model_eval_fn(model, batch_data):
        return _model_eval_fn(model, batch_data)
