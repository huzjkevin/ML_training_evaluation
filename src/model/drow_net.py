import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _conv1d_3
from .dr_spaam_fn import _model_fn, _model_eval_fn, _model_eval_collate_fn


class DrowNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(DrowNet, self).__init__()
        self.dropout = dropout

        self.conv_block_1 = nn.Sequential(
            _conv1d_3(1, 64), _conv1d_3(64, 64), _conv1d_3(64, 128)
        )
        self.conv_block_2 = nn.Sequential(
            _conv1d_3(128, 128), _conv1d_3(128, 128), _conv1d_3(128, 256)
        )
        self.conv_block_3 = nn.Sequential(
            _conv1d_3(256, 256), _conv1d_3(256, 256), _conv1d_3(256, 512)
        )
        self.conv_block_4 = nn.Sequential(_conv1d_3(512, 256), _conv1d_3(256, 128))

        self.conv_cls = nn.Conv1d(128, 1, kernel_size=1)
        self.conv_reg = nn.Conv1d(128, 2, kernel_size=1)

        # self.cls_loss = BinaryFocalLoss(gamma=focal_loss_gamma) \
        #             if focal_loss_gamma > 0.0 else F.binary_cross_entropy

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def model_fn(model, batch_data):
        return _model_fn(model, batch_data)

    @staticmethod
    def model_eval_fn(model, batch_data):
        return _model_eval_fn(model, batch_data)

    @staticmethod
    def model_eval_collate_fn(tb_dict_list, eval_dict_list):
        return _model_eval_collate_fn(tb_dict_list, eval_dict_list)

    def forward(self, x):
        """
        Args:
            x (tensor[B, CT, N, L]): (batch, cutout, scan, points per cutout)

        Returns:
            pred_cls (tensor[B, CT, C]): C = number of class
            pred_reg (tensor[B, CT, 2])
        """
        n_batch, n_cutout, n_scan, n_pts = x.shape

        # forward cutout from all scans
        out = x.view(n_batch * n_cutout * n_scan, 1, n_pts)
        out = self._conv_and_pool(out, self.conv_block_1)  # /2
        out = self._conv_and_pool(out, self.conv_block_2)  # /4

        # (batch, cutout, scan, channel, pts)
        out = out.view(n_batch, n_cutout, n_scan, out.shape[-2], out.shape[-1])
        # combine all scans
        out = torch.sum(out, dim=2)  # (B, CT, C, L)

        # forward fused cutout
        out = out.view(n_batch * n_cutout, out.shape[-2], out.shape[-1])
        out = self._conv_and_pool(out, self.conv_block_3)  # /8
        out = self.conv_block_4(out)
        out = F.avg_pool1d(out, kernel_size=out.shape[-1])  # (B * CT, C, 1)

        pred_cls = self.conv_cls(out).view(n_batch, n_cutout, -1)  # (B, CT, cls)
        pred_reg = self.conv_reg(out).view(n_batch, n_cutout, 2)  # (B, CT, 2)

        return pred_cls, pred_reg

    def _conv_and_pool(self, x, conv_block):
        out = conv_block(x)
        out = F.max_pool1d(out, kernel_size=2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out
