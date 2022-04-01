import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from dr_spaam.data_handle.jrdb_handle import JRDBHandle
import dr_spaam.utils.utils as u
import dr_spaam.utils.jrdb_transforms as jt


_XY_LIM = (-7, 7)
_Z_LIM = (-1, 2)
_INTERACTIVE = False
_SAVE_DIR = "/home/jia/tmp_imgs/test_jrdb_handle"


def _get_pts_color(pts, dim):
    d = np.clip(np.hypot(pts[0], pts[1]), 0.0, 10.0) / 10.0
    color = d.reshape(-1, 1).repeat(3, axis=1)
    color[:, dim] = 1
    return color


def _plot_sequence():
    jrdb_handle = JRDBHandle(
        split="train", cfg={"data_dir": "./data/JRDB", "num_scans": 5, "scan_stride": 1}
    )

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 2, figure=fig)

    ax_bev = fig.add_subplot(gs[:, 1])
    ax_fpv_xz = fig.add_subplot(gs[0, 0])
    ax_fpv_yz = fig.add_subplot(gs[1, 0])

    color_pool = np.random.uniform(size=(100, 3))

    _break = False

    if _INTERACTIVE:

        def p(event):
            nonlocal _break
            _break = True

        fig.canvas.mpl_connect("key_press_event", p)
    else:
        if os.path.exists(_SAVE_DIR):
            shutil.rmtree(_SAVE_DIR)
        os.makedirs(_SAVE_DIR)

    for i, data_dict in enumerate(jrdb_handle):
        if _break:
            break

        pc_xyz_upper = jt.transform_pts_upper_velodyne_to_base(
            data_dict["pc_data"]["upper_velodyne"]
        )
        pc_xyz_lower = jt.transform_pts_lower_velodyne_to_base(
            data_dict["pc_data"]["lower_velodyne"]
        )

        boxes, label_ids = [], []
        for ann in data_dict["anns"]:
            jrdb_handle.box_is_on_ground(ann)
            boxes.append(jt.box_from_jrdb(ann["box"]))
            label_ids.append(int(ann["label_id"].split(":")[-1]) % len(color_pool))

        laser_r = data_dict["laser_data"][-1]
        laser_phi = data_dict["laser_grid"]
        laser_z = data_dict["laser_z"]
        laser_x, laser_y = u.rphi_to_xy(laser_r, laser_phi)
        pc_xyz_laser = jt.transform_pts_laser_to_base(
            np.stack((laser_x, laser_y, laser_z), axis=0)
        )

        ax_bev.cla()
        ax_fpv_xz.cla()
        ax_fpv_yz.cla()

        ax_bev.set_aspect("equal")
        ax_bev.set_xlim(_XY_LIM[0], _XY_LIM[1])
        ax_bev.set_ylim(_XY_LIM[0], _XY_LIM[1])
        ax_bev.set_title(f"Frame {data_dict['idx']}. Press any key to exit.")
        ax_bev.set_xlabel("x [m]")
        ax_bev.set_ylabel("y [m]")
        # ax_bev.axis("off")

        for rgb_dim, pc_xyz in zip(
            (2, 1, 0), (pc_xyz_upper, pc_xyz_lower, pc_xyz_laser)
        ):
            ax_bev.scatter(pc_xyz[0], pc_xyz[1], s=1, c=_get_pts_color(pc_xyz, rgb_dim))
        for label_id, box in zip(label_ids, boxes):
            box.draw_bev(ax_bev, c=color_pool[label_id])

        for dim, ax_fpv in zip((0, 1), (ax_fpv_xz, ax_fpv_yz)):
            ax_fpv.set_aspect("equal")
            ax_fpv.set_xlim(_XY_LIM[0], _XY_LIM[1])
            ax_fpv.set_ylim(_Z_LIM[0], _Z_LIM[1])
            ax_fpv.set_title(f"Frame {data_dict['idx']}. Press any key to exit.")
            ax_fpv.set_xlabel("x [m]" if dim == 0 else "y [m]")
            ax_fpv.set_ylabel("z [m]")
            # ax_fpv.axis("off")

            for rgb_dim, pc_xyz in zip(
                (2, 1, 0), (pc_xyz_upper, pc_xyz_lower, pc_xyz_laser)
            ):
                ax_fpv.scatter(
                    pc_xyz[dim], pc_xyz[2], s=1, c=_get_pts_color(pc_xyz, rgb_dim)
                )
            for label_id, box in zip(label_ids, boxes):
                box.draw_fpv(ax_fpv, dim=dim, c=color_pool[label_id])

        if _INTERACTIVE:
            plt.pause(0.1)
        else:
            plt.savefig(os.path.join(_SAVE_DIR, f"frame_{data_dict['idx']:04}.png"))

    if _INTERACTIVE:
        plt.show()


if __name__ == "__main__":
    _plot_sequence()
