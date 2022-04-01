import numpy as np
import torch
import torch.nn.functional as F

import dr_spaam.utils.utils as u
import dr_spaam.utils.precision_recall as pru
from dr_spaam.utils.plotting import plot_one_batch_detr


def _get_localization_loss(pred, target, mask):
    """
    Args:
        pred (tensor[B, N, 2]):
        target (tensor[B, N, 2]):
        mask (tensor[B * N]): Compute loss using only masked elements
    """
    B, N = target.shape[:2]
    target = target.view(B * N, -1)
    pred = pred.view(B * N, -1)
    loc_loss = F.mse_loss(pred[mask], target[mask], reduction="none")
    loc_loss = torch.sqrt(torch.sum(loc_loss, dim=1)).mean()
    return loc_loss


def _model_fn(model, batch_dict):
    tb_dict, rtn_dict = {}, {}

    net_input = torch.from_numpy(batch_dict["input"]).cuda(non_blocking=True).float()

    pred_cls, pred_reg, pred_reg_prev = model(net_input)

    target_cls = (
        torch.from_numpy(batch_dict["target_cls"]).cuda(non_blocking=True).float()
    )
    target_reg = (
        torch.from_numpy(batch_dict["target_reg"]).cuda(non_blocking=True).float()
    )
    target_reg_prev = (
        torch.from_numpy(batch_dict["target_reg_prev"]).cuda(non_blocking=True).float()
    )
    target_tracking_flag = (
        torch.from_numpy(batch_dict["target_tracking_flag"])
        .cuda(non_blocking=True)
        .bool()
    )

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
    fg_mask = target_cls.view(-1).ne(0)
    fg_ratio = torch.sum(fg_mask).item() / (B * N)
    tb_dict["fg_ratio"] = fg_ratio

    # reg loss
    if fg_ratio > 0.0:
        reg_loss = _get_localization_loss(pred_reg, target_reg, fg_mask)
        total_loss = total_loss + reg_loss
        tb_dict["reg_loss"] = reg_loss.item()

    # reg loss for previous frame
    tracking_ratio = target_tracking_flag.sum().item() / (B * N)
    tb_dict["tracking_ratio"] = tracking_ratio

    if tracking_ratio > 0.0:
        reg_loss_prev = _get_localization_loss(
            pred_reg_prev, target_reg_prev, target_tracking_flag.view(-1)
        )
        total_loss = total_loss + reg_loss_prev
        tb_dict["reg_loss_prev"] = reg_loss_prev.item()

    rtn_dict["pred_reg"] = pred_reg.view(B, N, 2)
    rtn_dict["pred_cls"] = pred_cls.view(B, N)
    rtn_dict["pred_reg_prev"] = pred_reg_prev.view(B, N, 2)

    return total_loss, tb_dict, rtn_dict


def _model_eval_fn(model, batch_dict):
    _, tb_dict, rtn_dict = _model_fn(model, batch_dict)

    pred_cls = torch.sigmoid(rtn_dict["pred_cls"]).data.cpu().numpy()
    pred_reg = rtn_dict["pred_reg"].data.cpu().numpy()
    pred_reg_prev = rtn_dict["pred_reg_prev"].data.cpu().numpy()

    # # DEBUG use perfect predictions
    # pred_cls = batch_dict["target_cls"]
    # pred_cls[pred_cls < 0] = 1
    # pred_reg = batch_dict["target_reg"]

    # postprocess network prediction to get detection
    dets_xy_batch = []
    dets_xy_prev_batch = []
    dets_cls_batch = []
    dets_inds_batch = []
    anns_rphi_batch = []
    anns_rphi_prev_batch = []
    anns_tracking_mask_batch = []
    anns_inds_batch = []
    anns_valid_mask_batch = []
    for ib in range(pred_cls.shape[0]):
        scans = batch_dict["frame_dict_curr"][ib]["laser_data"]
        scan_phi = batch_dict["frame_dict_curr"][ib]["laser_grid"]

        # store detection, which will be used by _model_eval_collate_fn to compute AP
        dets_xy, dets_xy_prev, dets_cls, _ = u.nms_predicted_center(
            scans[-1], scan_phi, pred_cls[ib], pred_reg[ib], pred_reg_prev[ib]
        )
        if len(dets_xy) > 0:
            dets_xy_batch.append(dets_xy)
            dets_xy_prev_batch.append(dets_xy_prev)
            dets_cls_batch.append(dets_cls)
            dets_inds_batch = dets_inds_batch + [ib] * len(dets_cls)

        # store annotation
        anns_rphi = batch_dict["frame_dict_curr"][ib]["dets_rphi"].T
        anns_rphi_prev = batch_dict["frame_dict_curr"][ib]["dets_rphi_prev"].T
        if len(anns_rphi) > 0:
            anns_rphi_batch.append(anns_rphi)
            anns_rphi_prev_batch.append(anns_rphi_prev)
            anns_inds_batch = anns_inds_batch + [ib] * len(anns_rphi)
            anns_valid_mask_batch.append(batch_dict["anns_valid_mask"][ib])
            anns_tracking_mask_batch.append(batch_dict["anns_tracking_mask"][ib])

    if len(dets_xy_batch) > 0:
        dets_xy_batch = np.concatenate(dets_xy_batch, axis=0)
        dets_xy_prev_batch = np.concatenate(dets_xy_prev_batch, axis=0)
        dets_cls_batch = np.concatenate(dets_cls_batch, axis=0)
        dets_inds_batch = np.array(dets_inds_batch, dtype=np.int32)

    if len(anns_rphi_batch) > 0:
        anns_rphi_batch = np.concatenate(anns_rphi_batch, axis=0)
        anns_rphi_prev_batch = np.concatenate(anns_rphi_prev_batch, axis=0)
        anns_inds_batch = np.array(anns_inds_batch, dtype=np.int32)
        anns_valid_mask_batch = np.concatenate(anns_valid_mask_batch, axis=0)
        anns_tracking_mask_batch = np.concatenate(anns_tracking_mask_batch, axis=0)

    rtn_dict = {
        "dets_xy": dets_xy_batch,
        "dets_xy_prev": dets_xy_prev_batch,
        "dets_cls": dets_cls_batch,
        "dets_inds": dets_inds_batch,
        "anns_rphi": anns_rphi_batch,
        "anns_rphi_prev": anns_rphi_prev_batch,
        "anns_inds": anns_inds_batch,
        "anns_valid_mask": anns_valid_mask_batch,
        "anns_tracking_mask": anns_tracking_mask_batch,
    }

    # TODO assign a flag for plotting
    do_viz = False
    if do_viz:
        batch_dict["pred_cls"] = pred_cls
        batch_dict["pred_reg"] = pred_reg
        batch_dict["pred_reg_prev"] = pred_reg_prev
        inference_im = plot_one_batch_detr(batch_dict)
    else:
        inference_im = []
    fig_dict = {"inference_im": inference_im}

    return tb_dict, rtn_dict, fig_dict


def _model_eval_collate_fn(tb_dict_list, eval_dict_list):
    # collate tensorboard logs, taking the mean from all batches as the value
    # of the epoch
    epoch_tb_dict = {}
    for tb_dict in tb_dict_list:
        for k, v in tb_dict.items():
            epoch_tb_dict.setdefault(k, []).append(v)
    for k, v in epoch_tb_dict.items():
        epoch_tb_dict[k] = np.array(v).mean()

    # collate detections and annotations from all batches to evaluate epoch performance
    dets_xy_epoch = []
    dets_xy_prev_epoch = []
    dets_cls_epoch = []
    dets_inds_epoch = []
    anns_rphi_epoch = []
    anns_rphi_prev_epoch = []
    anns_inds_epoch = []
    anns_valid_mask_epoch = []
    anns_tracking_mask_epoch = []
    counter = 0
    for ie, eval_dict in enumerate(eval_dict_list):
        dets_inds_counter = 0
        if len(eval_dict["dets_xy"]) > 0:
            dets_xy_epoch.append(eval_dict["dets_xy"])
            dets_xy_prev_epoch.append(eval_dict["dets_xy_prev"])
            dets_cls_epoch.append(eval_dict["dets_cls"])
            dets_inds_epoch.append(eval_dict["dets_inds"] + counter)
            dets_inds_counter = eval_dict["dets_inds"].max() + counter

        anns_inds_counter = 0
        if len(eval_dict["anns_rphi"]) > 0:
            anns_rphi_epoch.append(eval_dict["anns_rphi"])
            anns_rphi_prev_epoch.append(eval_dict["anns_rphi_prev"])
            anns_inds_epoch.append(eval_dict["anns_inds"] + counter)
            anns_inds_counter = eval_dict["anns_inds"].max() + counter
            anns_valid_mask_epoch.append(eval_dict["anns_valid_mask"])
            anns_tracking_mask_epoch.append(eval_dict["anns_tracking_mask"])

        counter = max(anns_inds_counter, dets_inds_counter) + 1

    dets_xy_epoch = np.concatenate(dets_xy_epoch, axis=0)
    dets_xy_prev_epoch = np.concatenate(dets_xy_prev_epoch, axis=0)
    dets_cls_epoch = np.concatenate(dets_cls_epoch)
    dets_inds_epoch = np.concatenate(dets_inds_epoch)
    anns_rphi_epoch = np.concatenate(anns_rphi_epoch, axis=0)
    anns_rphi_prev_epoch = np.concatenate(anns_rphi_prev_epoch, axis=0)
    anns_inds_epoch = np.concatenate(anns_inds_epoch)
    anns_xy_epoch = np.stack(
        u.rphi_to_xy(anns_rphi_epoch[:, 0], anns_rphi_epoch[:, 1]), axis=1
    )
    anns_xy_prev_epoch = np.stack(
        u.rphi_to_xy(anns_rphi_prev_epoch[:, 0], anns_rphi_prev_epoch[:, 1]), axis=1
    )
    anns_valid_mask_epoch = np.concatenate(anns_valid_mask_epoch, axis=0)
    anns_tracking_mask_epoch = np.concatenate(anns_tracking_mask_epoch, axis=0)

    # ap
    epoch_dict = {}
    prev_valid_mask = np.logical_and(anns_tracking_mask_epoch, anns_valid_mask_epoch)
    for association_radius in (0.5, 0.3):
        # all annotation of current frame
        pr_dict = pru.get_precision_recall(
            dets_xy_epoch,
            dets_cls_epoch,
            dets_inds_epoch,
            anns_xy_epoch,
            anns_inds_epoch,
            association_radius,
        )
        for k, v in pr_dict.items():
            epoch_dict[f"{k}_curr_all_r{association_radius}"] = v

        # valid annotation of current frame
        pr_dict = pru.get_precision_recall(
            dets_xy_epoch,
            dets_cls_epoch,
            dets_inds_epoch,
            anns_xy_epoch[anns_valid_mask_epoch],
            anns_inds_epoch[anns_valid_mask_epoch],
            association_radius,
        )
        for k, v in pr_dict.items():
            epoch_dict[f"{k}_curr_valid_r{association_radius}"] = v
            epoch_dict[f"{k}_r{association_radius}"] = v  # to be consistent with DROW

        # all annotation of previous frame
        pr_dict = pru.get_precision_recall(
            dets_xy_prev_epoch,
            dets_cls_epoch,
            dets_inds_epoch,
            anns_xy_prev_epoch[anns_tracking_mask_epoch],
            anns_inds_epoch[anns_tracking_mask_epoch],
            association_radius,
        )
        for k, v in pr_dict.items():
            epoch_dict[f"{k}_prev_all_r{association_radius}"] = v

        # valid annotation of previous frame
        pr_dict = pru.get_precision_recall(
            dets_xy_prev_epoch,
            dets_cls_epoch,
            dets_inds_epoch,
            anns_xy_prev_epoch[prev_valid_mask],
            anns_inds_epoch[prev_valid_mask],
            association_radius,
        )
        for k, v in pr_dict.items():
            epoch_dict[f"{k}_prev_valid_r{association_radius}"] = v

    # log to tensorboard ap, peak_f1, and eer
    for k, v in epoch_dict.items():
        if not isinstance(v, (np.ndarray, list, tuple)):
            epoch_tb_dict[k] = v

    return epoch_tb_dict, epoch_dict
