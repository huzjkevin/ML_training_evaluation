import numpy as np
from torch.utils.data import Dataset

from .jrdb_handle import JRDBHandle
import dr_spaam.utils.utils as u
import dr_spaam.utils.jrdb_transforms as jt


class JRDBDataset(Dataset):
    def __init__(self, split, cfg):
        self.__handle = JRDBHandle(split, cfg["DataHandle"])

        self._augment_data = cfg["augment_data"]
        self._person_only = cfg["person_only"]
        self._cutout_kwargs = cfg["cutout_kwargs"]

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        data_dict = self.__handle[idx]

        # get annotation in laser frame
        ann_xyz = []
        for ann in data_dict["anns"]:
            ann_xyz.append([ann["box"]["cx"], ann["box"]["cy"], ann["box"]["cz"]])
        ann_xyz = np.array(ann_xyz, dtype=np.float32).T
        ann_xyz = jt.transform_pts_base_to_laser(ann_xyz)

        # DROW defines laser frame as x-forward, y-right, z-downward
        # JRDB defines laser frame as x-forward, y-left, z-upward
        # Training code follows DROW frame convention
        ann_xyz[1] = -ann_xyz[1]
        # equivalent of inversing laser phi angle
        data_dict["laser_data"] = data_dict["laser_data"][:, ::-1]

        # regression target
        scan_rphi = np.stack(
            (data_dict["laser_data"][-1], data_dict["laser_grid"]), axis=0
        )
        dets_rphi = np.stack(u.xy_to_rphi(ann_xyz[0], ann_xyz[1]), axis=0)

        target_cls, target_reg, anns_valid_mask = _get_regression_target(
            scan_rphi,
            dets_rphi,
            person_radius_small=0.4,
            person_radius_large=0.8,
            min_close_points=5,
        )

        data_dict["target_cls"] = target_cls
        data_dict["target_reg"] = target_reg
        data_dict["anns_valid_mask"] = anns_valid_mask

        # to be consistent with DROWDataset in order to use the same evaluation function
        dets_wp = []
        for i in range(dets_rphi.shape[1]):
            dets_wp.append((dets_rphi[0, i], dets_rphi[1, i]))
        data_dict["dets_wp"] = dets_wp
        data_dict["scans"] = data_dict["laser_data"]
        data_dict["scan_phi"] = data_dict["laser_grid"]

        if self._augment_data:
            data_dict = u.data_augmentation(data_dict)

        data_dict["input"] = u.scans_to_cutout(
            data_dict["laser_data"],
            data_dict["laser_grid"],
            stride=1,
            **self._cutout_kwargs
        )

        return data_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["target_cls", "target_reg", "input"]:
                rtn_dict[k] = np.array([sample[k] for sample in batch])
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict


def _get_regression_target(
    scan_rphi, dets_rphi, person_radius_small, person_radius_large, min_close_points
):
    """Generate classification and regression label.

    Args:
        scan_rphi (np.ndarray[2, N]): Scan points in polar coordinate
        dets_rphi (np.ndarray[2, M]): Annotated person centers in polar coordinate
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
    """
    N = scan_rphi.shape[1]

    scan_xy = np.stack(u.rphi_to_xy(scan_rphi[0], scan_rphi[1]), axis=0)
    dets_xy = np.stack(u.rphi_to_xy(dets_rphi[0], dets_rphi[1]), axis=0)

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
    dets_matched_rphi = dets_rphi[:, argmin_dist_scan_dets]
    target_reg = np.stack(
        u.global_to_canonical(
            scan_rphi[0], scan_rphi[1], dets_matched_rphi[0], dets_matched_rphi[1]
        ),
        axis=1,
    )

    return target_cls, target_reg, anns_valid_mask
