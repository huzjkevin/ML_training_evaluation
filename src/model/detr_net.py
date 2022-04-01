from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _conv1d_3
from .detr_net_fn import _model_fn, _model_eval_fn, _model_eval_collate_fn


class DeTrNet(nn.Module):
    def __init__(self, dropout=0.5, num_pts=56, embedding_length=128, window_size=11):
        super(DeTrNet, self).__init__()
        self._dropout = dropout
        self._embedding_length = embedding_length
        self._window_size = window_size

        # backbone
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

        # layers converting features to embedding for computing correlation
        self.to_embedding = nn.Sequential(
            nn.Conv1d(
                256,
                self._embedding_length,
                kernel_size=int(ceil(num_pts / 4)),
                padding=0,
            ),
            nn.BatchNorm1d(self._embedding_length),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # correlation
        self.corr_net = _CorrelationNet(window_size=window_size, panoramatic_scan=True)

        # detection layer
        self.conv_cls = nn.Conv1d(128, 1, kernel_size=1)
        self.conv_reg = nn.Conv1d(128, 2, kernel_size=1)
        self.conv_reg_prev = nn.Conv1d(128 + window_size, 2, kernel_size=1)

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

    def forward(self, x, inference=False):
        """
        Args:
            x (tensor[B, CT, N, L]): (batch, cutout, scan, points per cutout)
            inference (bool, optional): Set to true for sequencial inference
                (i.e. in deployment). Defaults to False.

        Returns:
            pred_cls (tensor[B, CT, C]): C = number of class
            pred_reg (tensor[B, CT, 2])
        """
        B, CT, N, L = x.shape

        # TODO save previous embedding and feature when doing inference

        # process two scans
        out_0, emb_0 = self._forward_backbone(x[:, :, 0, :])
        out_1, emb_1 = self._forward_backbone(x[:, :, 1, :])

        corr = self.corr_net(
            emb_0.view(B, CT, self._embedding_length),
            emb_1.view(B, CT, self._embedding_length),
        )  # (B, CT, W)

        pred_cls = self.conv_cls(out_1).view(B, CT, -1)
        pred_reg = self.conv_reg(out_1).view(B, CT, 2)

        reg_prev_feature = torch.cat(
            (out_1, corr.view(B * CT, self._window_size, 1)), dim=1
        )
        pred_reg_prev = self.conv_reg_prev(reg_prev_feature).view(B, CT, 2)

        return pred_cls, pred_reg, pred_reg_prev

    def _forward_backbone(self, x):
        """Forward backbone network for one scan

        Args:
            x (tensor[B, CT, L]): A single scan
        """
        B, CT, L = x.shape

        out = x.view(B * CT, 1, L)
        out = self._conv_and_pool(out, self.conv_block_1)  # /2
        out = self._conv_and_pool(out, self.conv_block_2)  # /4

        # embedding for computing correlation
        embedding = self.to_embedding(out)  # (B * CT, embedding_length, 1)

        out = self._conv_and_pool(out, self.conv_block_3)  # /8
        out = self.conv_block_4(out)
        out = F.avg_pool1d(out, kernel_size=out.shape[-1])  # (B * CT, C, 1)

        return out, embedding

    def _conv_and_pool(self, x, conv_block):
        out = conv_block(x)
        out = F.max_pool1d(out, kernel_size=2)
        if self._dropout > 0:
            out = F.dropout(out, p=self._dropout, training=self.training)

        return out


class _CorrelationNet(nn.Module):
    def __init__(self, window_size, panoramatic_scan):
        """Learnable correlation between two scans.

        Args:
            window_size (int): Full neighborhood window size to compute attention
            panoramatic_scan (bool): True if the scan covers full 360 degree
        """
        super(_CorrelationNet, self).__init__()
        self._window_size = window_size
        self._panoramatic_scan = panoramatic_scan

        # place holder, created at runtime
        self.neighbor_inds = None

        # TODO Introduce a learnable correlation kernel
        # https://arxiv.org/pdf/1906.03349.pdf

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x0, x1):
        """
        Args:
            x0 (tensor[B, CT, C]): Feature embedding for previous scan
                (batch, cutout, channel)
            x1 (tensor): For current scan

        Returns:
            tensor: Correlation matrix
        """
        B, CT, C = x0.shape

        # only need to generate neighbor mask once
        if self.neighbor_inds is None:
            self.neighbor_inds = self._generate_neighbor_indices(x0)

        # pair-wise correlation
        # TODO This may be optimized, only need to compute correlation for nearby points
        corr = torch.matmul(x1, x0.permute(0, 2, 1)) / float(C)  # (B, CT, CT)
        corr = torch.gather(corr, 2, self.neighbor_inds)  # (B, CT, W)

        return corr

    def _generate_neighbor_indices(self, x):
        """Generate neighborhood indices

        Args:
            x (tensor): See `forward()` method

        Returns:
            inds_window (tensor[B, CT, W]): (i, j) element represents for the ith point,
                the index of its jth neighboring points. This matrix is replicated
                along batch dimension.
        """
        # indices of neighboring cutout
        hw = int(self._window_size / 2)
        window_inds = torch.arange(-hw, hw + 1).long()  # (W,)

        CT = x.shape[1]
        inds_window = torch.arange(CT).unsqueeze(dim=-1).long()  # (CT, 1)
        inds_window = inds_window + window_inds.unsqueeze(dim=0)  # (CT, W)
        if self._panoramatic_scan:
            inds_window = inds_window % CT
        else:
            inds_window = inds_window.clamp(min=0, max=CT - 1)

        inds_window = inds_window.repeat(x.shape[0], 1, 1)

        return inds_window.cuda(x.get_device()) if x.is_cuda else inds_window
