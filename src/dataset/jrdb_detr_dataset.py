from .jrdb_handle import JRDBHandle

import numpy as np
from torch.utils.data import Dataset

import dr_spaam.utils.utils as u
import dr_spaam.utils.jrdb_transforms as jt


class JRDBDeTrDataset(Dataset):
    def __init__(self, split, cfg):
        self.__handle = JRDBHandle(split, cfg["DataHandle"])
        self._cutout_kwargs = cfg["cutout_kwargs"]

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        frame_dict_curr = self.__handle[idx]
        if not frame_dict_curr["first_frame"]:
            frame_dict_prev = self.__handle[idx - 1]
        else:
            frame_dict_prev = frame_dict_curr
            frame_dict_curr = self.__handle[idx + 1]

        # get matching annotations in JRDB base frame
        ann_xyz_curr = []
        ann_xyz_prev = []
        match_found = []
        for ann_curr in frame_dict_curr["anns"]:
            ann_xyz_curr.append(
                [ann_curr["box"]["cx"], ann_curr["box"]["cy"], ann_curr["box"]["cz"]]
            )

            # find matching annotation in previous frame
            match_found.append(False)
            for ann_prev in frame_dict_prev["anns"]:
                if ann_prev["label_id"] != ann_curr["label_id"]:
                    continue
                match_found[-1] = True
                ann_xyz_prev.append(
                    [
                        ann_prev["box"]["cx"],
                        ann_prev["box"]["cy"],
                        ann_prev["box"]["cz"],
                    ]
                )

            if not match_found[-1]:
                ann_xyz_prev.append([0, 0, 0])  # place holder

        # convert annotations to JRDB laser frame
        ann_xyz_prev = np.array(ann_xyz_prev, dtype=np.float32).T
        ann_xyz_prev = jt.transform_pts_base_to_laser(ann_xyz_prev)
        ann_xyz_curr = np.array(ann_xyz_curr, dtype=np.float32).T
        ann_xyz_curr = jt.transform_pts_base_to_laser(ann_xyz_curr)

        # DROW defines laser frame as x-forward, y-right, z-downward
        # JRDB defines laser frame as x-forward, y-left, z-upward
        # Training code follows DROW frame convention
        ann_xyz_prev[1] = -ann_xyz_prev[1]
        ann_xyz_curr[1] = -ann_xyz_curr[1]

        # equivalent of inversing laser phi angle
        frame_dict_prev["laser_data"] = frame_dict_prev["laser_data"][:, ::-1]
        frame_dict_curr["laser_data"] = frame_dict_curr["laser_data"][:, ::-1]

        # annotations
        scan_rphi = np.stack(
            (frame_dict_curr["laser_data"][-1], frame_dict_curr["laser_grid"]), axis=0
        )
        dets_rphi_prev = np.stack(
            u.xy_to_rphi(ann_xyz_prev[0], ann_xyz_prev[1]), axis=0
        )
        dets_rphi_curr = np.stack(
            u.xy_to_rphi(ann_xyz_curr[0], ann_xyz_curr[1]), axis=0
        )

        frame_dict_curr["dets_rphi_prev"] = dets_rphi_prev
        frame_dict_curr["dets_rphi"] = dets_rphi_curr

        # network target
        match_found = np.array(match_found, dtype=np.bool)
        (
            target_cls,
            target_reg,
            anns_valid_mask,
            target_reg_prev,
            target_tracking_flag,
        ) = _get_detr_target(
            scan_rphi,
            dets_rphi_prev,
            dets_rphi_curr,
            match_found,
            person_radius_small=0.4,
            person_radius_large=0.8,
            min_close_points=5,
        )

        frame_dict_combined = {
            "frame_dict_prev": frame_dict_prev,
            "frame_dict_curr": frame_dict_curr,
            "target_cls": target_cls,
            "target_reg": target_reg,
            "target_reg_prev": target_reg_prev,
            "target_tracking_flag": target_tracking_flag,
            "anns_valid_mask": anns_valid_mask,
            "anns_tracking_mask": match_found,
        }

        # cutout
        scans = np.stack(
            (frame_dict_prev["laser_data"][-1], frame_dict_curr["laser_data"][-1]),
            axis=0,
        )

        frame_dict_combined["input"] = u.scans_to_cutout(
            scans, frame_dict_curr["laser_grid"], stride=1, **self._cutout_kwargs
        )

        return frame_dict_combined

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in [
                "target_cls",
                "target_reg",
                "target_reg_prev",
                "target_tracking_flag",
                "input",
            ]:
                rtn_dict[k] = np.array([sample[k] for sample in batch])
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict


def _get_detr_target(
    scan_rphi,
    dets_rphi_prev,
    dets_rphi_curr,
    match_found_flag,
    person_radius_small,
    person_radius_large,
    min_close_points,
):
    """Generate cls, reg, and offset reg for the DeTr network.

    Args:
        scan_rphi (np.ndarray[2, N]): Scan points in polar coordinate of the current
            frame
        dets_rphi_prev (np.ndarray[2, M]): Annotated person centers in polar coordinate
            of the previous frame
        dets_rphi_curr (np.ndarray[2, M]): Of the current frame
        match_found_flag (np.ndarray[M]): True if the ith detection in the current
            frame has matching in the previous frame (i.e. it is not a new person)
        person_radius_small (float): Points less than this distance away
            from an annotation is assigned to that annotation and marked as fg.
        person_radius_large (float): Points with no annotation smaller
            than this distance is marked as bg.
        min_close_points (int): Annotations with supportive points fewer than this
            value is marked as invalid. Supportive points are those within the small
            radius.

    Returns:
        target_cls (np.ndarray[N]): Classification label, 1=fg, 0=bg, -1=ignore
        target_reg (np.ndarray[N, 2]): Regression label
        anns_valid_mask (np.ndarray[M])
        target_reg_prev (np.ndarray[N, 2]): Regression label for previous frame
        target_tracking_flag (np.ndarray[N]): Only supervise tracking on points
            with this flag
    """
    N = scan_rphi.shape[1]

    scan_xy = np.stack(u.rphi_to_xy(scan_rphi[0], scan_rphi[1]), axis=0)
    dets_xy = np.stack(u.rphi_to_xy(dets_rphi_curr[0], dets_rphi_curr[1]), axis=0)

    dist_scan_dets = np.hypot(
        scan_xy[0].reshape(1, -1) - dets_xy[0].reshape(-1, 1),
        scan_xy[1].reshape(1, -1) - dets_xy[1].reshape(-1, 1),
    )  # (M, N) pairwise distance between scan and detections

    # mark out annotations that has too few scan points
    anns_valid_mask = (
        np.sum(dist_scan_dets < person_radius_small, axis=1) > min_close_points
    )  # (M, )

    # for each point, find the distance to its closest annotation
    argmin_dist_scan_dets = np.argmin(dist_scan_dets, axis=0)  # (N, )
    min_dist_scan_dets = dist_scan_dets[argmin_dist_scan_dets, np.arange(N)]

    # points within small radius, whose corresponding annotation is valid, is marked
    # as foreground
    target_cls = -1 * np.ones(N, dtype=np.int64)
    fg_mask = np.logical_and(
        anns_valid_mask[argmin_dist_scan_dets], min_dist_scan_dets < person_radius_small
    )
    target_cls[fg_mask] = 1
    target_cls[min_dist_scan_dets > person_radius_large] = 0

    # regression target
    dets_matched_rphi = dets_rphi_curr[:, argmin_dist_scan_dets]
    target_reg = np.stack(
        u.global_to_canonical(
            scan_rphi[0], scan_rphi[1], dets_matched_rphi[0], dets_matched_rphi[1]
        ),
        axis=1,
    )

    # tracking target
    dets_xy_prev = np.stack(u.rphi_to_xy(dets_rphi_prev[0], dets_rphi_prev[1]), axis=0)
    dets_rphi_prev = np.stack(u.xy_to_rphi(dets_xy_prev[0], dets_xy_prev[1]), axis=0)
    dets_rphi_prev = dets_rphi_prev[:, argmin_dist_scan_dets]

    target_reg_prev = np.stack(
        u.global_to_canonical(
            scan_rphi[0], scan_rphi[1], dets_rphi_prev[0], dets_rphi_prev[1]
        ),
        axis=1,
    )

    target_tracking_flag = match_found_flag[argmin_dist_scan_dets]
    target_tracking_flag = np.logical_and(target_tracking_flag, target_cls != 0)

    return (
        target_cls,
        target_reg,
        anns_valid_mask,
        target_reg_prev,
        target_tracking_flag,
    )
