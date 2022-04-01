import numpy as np

import torch
import torch.nn.functional as F

import dr_spaam.utils.utils as u
import dr_spaam.utils.precision_recall as pru
from dr_spaam.utils.plotting import plot_one_batch


def _model_fn(model, batch_dict):
    tb_dict, rtn_dict = {}, {}

    net_input = batch_dict["input"]
    net_input = torch.from_numpy(net_input).cuda(non_blocking=True).float()

    rtn_tuple = model(net_input)

    # so this function can be used for both DROW and DR-SPAAM
    if len(rtn_tuple) == 2:
        pred_cls, pred_reg = rtn_tuple
    elif len(rtn_tuple) == 3:
        pred_cls, pred_reg, pred_sim = rtn_tuple
        rtn_dict["pred_sim"] = pred_sim

    target_cls, target_reg = batch_dict["target_cls"], batch_dict["target_reg"]
    target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).float()
    target_reg = torch.from_numpy(target_reg).cuda(non_blocking=True).float()

    # number of valid points
    B, N = target_cls.shape[:2]
    valid_mask = target_cls.view(-1).ge(0)
    valid_ratio = torch.sum(valid_mask).item() / (B * N)
    tb_dict["valid_ratio"] = valid_ratio
    assert valid_ratio > 0, "No valid points in this batch."

    # cls loss
    cls_loss = F.binary_cross_entropy_with_logits(
        pred_cls.view(-1)[valid_mask], target_cls.view(-1)[valid_mask], reduction="mean"
    )
    total_loss = cls_loss
    tb_dict["cls_loss"] = cls_loss.item()

    # number fg points
    # NOTE supervise regression for both close and far neighbor points
    fg_mask = target_cls.view(-1).ne(0)
    fg_ratio = torch.sum(fg_mask).item() / (B * N)
    tb_dict["fg_ratio"] = fg_ratio

    # reg loss
    if fg_ratio > 0.0:
        target_reg = target_reg.view(B * N, -1)
        pred_reg = pred_reg.view(B * N, -1)
        reg_loss = F.mse_loss(pred_reg[fg_mask], target_reg[fg_mask], reduction="none")
        reg_loss = torch.sqrt(torch.sum(reg_loss, dim=1)).mean()
        total_loss = total_loss + reg_loss
        tb_dict["reg_loss"] = reg_loss.item()

    # # regularization loss for spatial attention
    # if spatial_drow:
    #     # shannon entropy
    #     att_loss = (-torch.log(pred_sim + 1e-5) * pred_sim).sum(dim=2).mean()
    #     tb_dict['att_loss'] = att_loss.item()
    #     total_loss = total_loss + att_loss

    rtn_dict["pred_reg"] = pred_reg.view(B, N, 2)
    rtn_dict["pred_cls"] = pred_cls.view(B, N)

    return total_loss, tb_dict, rtn_dict


def _model_eval_fn(model, batch_dict):
    _, tb_dict, rtn_dict = _model_fn(model, batch_dict)

    pred_cls = torch.sigmoid(rtn_dict["pred_cls"]).data.cpu().numpy()
    pred_reg = rtn_dict["pred_reg"].data.cpu().numpy()

    # # DEBUG use perfect predictions
    # pred_cls = batch_dict["target_cls"]
    # pred_cls[pred_cls < 0] = 1
    # pred_reg = batch_dict["target_reg"]

    # postprocess network prediction to get detection
    scans = batch_dict["scans"]
    scan_phi = batch_dict["scan_phi"]
    dets_xy_batch = []
    dets_cls_batch = []
    dets_inds_batch = []
    anns_rphi_batch = []
    anns_inds_batch = []
    anns_valid_mask_batch = []
    for ib in range(len(scans)):
        # store detection, which will be used by _model_eval_collate_fn to compute AP
        dets_xy, dets_cls, _ = u.nms_predicted_center(
            scans[ib][-1], scan_phi[ib], pred_cls[ib], pred_reg[ib]
        )
        if len(dets_xy) > 0:
            dets_xy_batch.append(dets_xy)
            dets_cls_batch.append(dets_cls)
            dets_inds_batch = dets_inds_batch + [ib] * len(dets_cls)

        # store annotation
        anns_rphi = batch_dict["dets_wp"][ib]
        if len(anns_rphi) > 0:
            anns_rphi_batch.append(np.array(anns_rphi))
            anns_inds_batch = anns_inds_batch + [ib] * len(anns_rphi)
            anns_valid_mask_batch.append(batch_dict["anns_valid_mask"][ib])

    if len(dets_xy_batch) > 0:
        dets_xy_batch = np.concatenate(dets_xy_batch, axis=0)
        dets_cls_batch = np.concatenate(dets_cls_batch, axis=0)
        dets_inds_batch = np.array(dets_inds_batch, dtype=np.int32)

    if len(anns_rphi_batch) > 0:
        anns_rphi_batch = np.concatenate(anns_rphi_batch, axis=0)
        anns_inds_batch = np.array(anns_inds_batch, dtype=np.int32)
        anns_valid_mask_batch = np.concatenate(anns_valid_mask_batch, axis=0)

    rtn_dict = {
        "dets_xy": dets_xy_batch,
        "dets_cls": dets_cls_batch,
        "dets_inds": dets_inds_batch,
        "anns_rphi": anns_rphi_batch,
        "anns_inds": anns_inds_batch,
        "anns_valid_mask": anns_valid_mask_batch,
    }

    # TODO assign a flag for plotting
    do_viz = False
    if do_viz:
        batch_dict["pred_cls"] = pred_cls
        batch_dict["pred_reg"] = pred_reg
        inference_im = plot_one_batch(batch_dict)
    else:
        inference_im = []
    fig_dict = {"inference_im": inference_im}

    return tb_dict, rtn_dict, fig_dict


def _model_eval_collate_fn(tb_dict_list, eval_dict_list):
    # tb_dict should only contain scalar values, collate them into an array
    # and take their mean as the value of the epoch
    epoch_tb_dict = {}
    for tb_dict in tb_dict_list:
        for k, v in tb_dict.items():
            epoch_tb_dict.setdefault(k, []).append(v)
    for k, v in epoch_tb_dict.items():
        epoch_tb_dict[k] = np.array(v).mean()

    # collate detections and annotations from all batches to evaluate epoch performance
    dets_xy_epoch = []
    dets_cls_epoch = []
    dets_inds_epoch = []
    anns_rphi_epoch = []
    anns_inds_epoch = []
    anns_valid_mask_epoch = []
    counter = 0
    for ie, eval_dict in enumerate(eval_dict_list):
        dets_inds_counter = 0
        if len(eval_dict["dets_xy"]) > 0:
            dets_xy_epoch.append(eval_dict["dets_xy"])
            dets_cls_epoch.append(eval_dict["dets_cls"])
            dets_inds_epoch.append(eval_dict["dets_inds"] + counter)
            dets_inds_counter = eval_dict["dets_inds"].max() + counter

        anns_inds_counter = 0
        if len(eval_dict["anns_rphi"]) > 0:
            anns_rphi_epoch.append(eval_dict["anns_rphi"])
            anns_inds_epoch.append(eval_dict["anns_inds"] + counter)
            anns_inds_counter = eval_dict["anns_inds"].max() + counter
            anns_valid_mask_epoch.append(eval_dict["anns_valid_mask"])

        counter = max(anns_inds_counter, dets_inds_counter) + 1

    dets_xy_epoch = np.concatenate(dets_xy_epoch, axis=0)
    dets_cls_epoch = np.concatenate(dets_cls_epoch)
    dets_inds_epoch = np.concatenate(dets_inds_epoch)
    anns_rphi_epoch = np.concatenate(anns_rphi_epoch, axis=0)
    anns_inds_epoch = np.concatenate(anns_inds_epoch)
    anns_xy_epoch = np.stack(
        u.rphi_to_xy(anns_rphi_epoch[:, 0], anns_rphi_epoch[:, 1]), axis=1
    )
    anns_valid_mask_epoch = np.concatenate(anns_valid_mask_epoch, axis=0)

    # evaluate using only valid annotation
    anns_xy_epoch = anns_xy_epoch[anns_valid_mask_epoch]
    anns_inds_epoch = anns_inds_epoch[anns_valid_mask_epoch]

    # evaluate epoch
    epoch_dict = {}
    for association_radius in (0.5, 0.3):
        pr_dict = pru.get_precision_recall(
            dets_xy_epoch,
            dets_cls_epoch,
            dets_inds_epoch,
            anns_xy_epoch,
            anns_inds_epoch,
            association_radius,
        )
        for k, v in pr_dict.items():
            epoch_dict[f"{k}_r{association_radius}"] = v

    # log to tensorboard ap, peak_f1, and eer
    for k, v in epoch_dict.items():
        if not isinstance(v, (np.ndarray, list, tuple)):
            epoch_tb_dict[k] = v

    return epoch_tb_dict, epoch_dict
