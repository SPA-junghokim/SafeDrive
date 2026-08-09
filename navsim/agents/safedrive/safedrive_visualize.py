"""Side-by-side ground truth vs. prediction plots for SafeDrive.

Runs a checkpoint over cached samples and writes one figure per sample showing
BEV segmentation, detection, agent motion and the planned trajectory, each with
the ground truth next to the prediction.

    python navsim/agents/safedrive/safedrive_visualize.py \
        --cache exp/smoke_feature_cache --ckpt ckpts/safedrive_phase3_10ep.ckpt --out exp/viz

Tensor layouts, all verified against the training code rather than assumed:
  target agent_states      (N, 7)     x, y, yaw, length, width, vx, vy
  target agent_labels      (N,)       -1 padding, 0 vehicle, 1 pedestrian
  target motion_traj       (N, 9, 7)  index 0 is the current pose, 1: the future
  pred   ins_states_L      (Q, 8)     x, y, sin, cos, length, width, vx, vy
  pred   ins_labels_L      (Q,)       logit, apply sigmoid
  pred   agent_motion_traj_L (Q, 1, 8, 3)
      A DISPLACEMENT from each agent's own current position, not an ego-frame
      position: _motion_loss_single trains it against (gt_traj - gt_traj[0]).
      Add the predicted box centre before plotting.
"""
import argparse
import glob
import gzip
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.patches import Polygon as MplPolygon
from omegaconf import OmegaConf

from navsim.planning.script.run_training import custom_collate_fn

# BEV geometry: pixel = (x / 0.125, y / 0.125 + 256); rows are +x (forward), cols +y (left)
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.0, 64.0, -32.0, 32.0
EXTENT = [Y_MIN, Y_MAX, X_MIN, X_MAX]

CLASS_NAMES = ["background", "drivable", "walkway", "centerline", "static", "vehicle", "pedestrian"]
BEV_COLORS = np.array([
    [255, 255, 255], [110, 110, 110], [74, 191, 116], [255, 179, 0],
    [255, 0, 0], [0, 102, 255], [255, 0, 255],
], dtype=np.uint8)

GT_COLOR, PRED_COLOR = "#1a9850", "#d73027"


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def load_cached_sample(sample_dir):
    feature = pickle.load(gzip.open(os.path.join(sample_dir, "transfuser_feature.gz"), "rb"))
    target = pickle.load(gzip.open(os.path.join(sample_dir, "transfuser_target.gz"), "rb"))
    return feature, target


def colorize_bev(label_map):
    label_map = np.asarray(label_map).astype(np.int64)
    return BEV_COLORS[np.clip(label_map, 0, len(BEV_COLORS) - 1)]


def gt_boxes_from_targets(targets):
    """-> boxes (N, 5) x,y,yaw,length,width and the validity mask over the padded slots."""
    labels = targets["agent_labels"]
    valid = labels >= 0                     # -1 marks padding
    return targets["agent_states"][valid][:, :5], valid


def pred_boxes_from_states(states):
    """pred (Q, 8) x,y,sin,cos,length,width,... -> (Q, 5) x,y,yaw,length,width."""
    yaw = np.arctan2(states[:, 2], states[:, 3])
    return np.stack([states[:, 0], states[:, 1], yaw, states[:, 4], states[:, 5]], axis=-1)


def per_class_iou(gt, pred, num_classes=len(CLASS_NAMES)):
    ious = []
    for c in range(num_classes):
        g, p = gt == c, pred == c
        union = np.logical_or(g, p).sum()
        ious.append(np.logical_and(g, p).sum() / union if union else np.nan)
    return np.array(ious, dtype=float)


def draw_boxes(ax, boxes, color, label, linewidth=1.3):
    """boxes: (N, 5) x, y, yaw, length, width. Axes are (y, x)."""
    for i, (x, y, yaw, length, width) in enumerate(boxes):
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                            [-length / 2, -width / 2], [-length / 2, width / 2]])
        pts = corners @ np.array([[c, -s], [s, c]]).T + np.array([x, y])
        ax.add_patch(MplPolygon(pts[:, ::-1], closed=True, fill=False, edgecolor=color,
                                linewidth=linewidth, label=label if i == 0 else None))


def setup_bev_ax(ax, title, background=None):
    if background is not None:
        ax.imshow(background, extent=EXTENT, origin="lower", interpolation="nearest")
    ax.set_xlim(Y_MAX, Y_MIN)               # +y (left) drawn on the left
    ax.set_ylim(X_MIN, X_MAX)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("y [m]", fontsize=7)
    ax.set_ylabel("x [m]", fontsize=7)
    ax.tick_params(labelsize=6)


def camera_image(camera_feature):
    """Newest frame of the cached camera tensor, cameras stacked horizontally."""
    cam = np.asarray(camera_feature)
    while cam.ndim > 4:
        cam = cam[-1]
    img = (np.concatenate([c.transpose(1, 2, 0) for c in cam], axis=1) if cam.ndim == 4
           else cam.transpose(1, 2, 0))
    if img.max() <= 1.0:
        img = img * 255.0
    return img.astype(np.uint8)


def plot_sample(cam_img, token, gt, pred, score_thr):
    """Build the 3x3 comparison figure. gt/pred are the dicts assembled in main()."""
    fig = plt.figure(figsize=(17, 11))
    grid = fig.add_gridspec(3, 3, height_ratios=[0.85, 1.35, 1.35])
    gt_bg = colorize_bev(gt["bev"])

    ax = fig.add_subplot(grid[0, :])
    ax.imshow(cam_img)
    ax.set_title(f"camera   |   token {token}", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(grid[1, 0])
    setup_bev_ax(ax, "BEV segmentation - GT", background=gt_bg)

    ax = fig.add_subplot(grid[1, 1])
    setup_bev_ax(ax, "BEV segmentation - prediction", background=colorize_bev(pred["bev"]))

    ax = fig.add_subplot(grid[1, 2])
    iou = pred["iou"]
    shown = np.arange(len(iou))[~np.isnan(iou)]
    ax.barh(shown, iou[shown], color=[BEV_COLORS[c] / 255 for c in shown],
            edgecolor="k", linewidth=0.5)
    ax.set_yticks(shown)
    ax.set_yticklabels([CLASS_NAMES[c] for c in shown], fontsize=7)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(labelsize=6)
    ax.set_title(f"BEV seg IoU per class   (mean {np.nanmean(iou):.3f})", fontsize=9)

    ax = fig.add_subplot(grid[2, 0])
    setup_bev_ax(ax, f"detection   GT {len(gt['boxes'])} vs pred {len(pred['boxes'])} "
                     f"(score > {score_thr})", background=gt_bg)
    draw_boxes(ax, gt["boxes"], GT_COLOR, "GT")
    draw_boxes(ax, pred["boxes"], PRED_COLOR, "pred")
    ax.legend(fontsize=6, loc="upper right")

    ax = fig.add_subplot(grid[2, 1])
    setup_bev_ax(ax, "agent motion (8 future steps)", background=gt_bg)
    first = True
    for traj, mask, box in zip(gt["motion"], gt["motion_mask"], gt["boxes"]):
        if mask.any():
            pts = np.vstack([box[None, :2], traj[mask]])        # start at the box centre
            ax.plot(pts[:, 1], pts[:, 0], color=GT_COLOR, lw=1.1, label="GT" if first else None)
            first = False
    if pred["motion"] is not None:
        for i, (traj, box) in enumerate(zip(pred["motion"], pred["boxes"])):
            pts = np.vstack([box[None, :2], traj])
            ax.plot(pts[:, 1], pts[:, 0], color=PRED_COLOR, lw=1.1, ls="--",
                    label="pred" if i == 0 else None)
    ax.legend(fontsize=6, loc="upper right")

    ax = fig.add_subplot(grid[2, 2])
    setup_bev_ax(ax, f"planning   ADE {pred['ade']:.2f} m / FDE {pred['fde']:.2f} m",
                 background=gt_bg)
    ax.plot(np.r_[0, gt["plan"][:, 1]], np.r_[0, gt["plan"][:, 0]], "-o",
            color=GT_COLOR, ms=4, lw=2.0, label="GT")
    ax.plot(np.r_[0, pred["plan"][:, 1]], np.r_[0, pred["plan"][:, 0]], "--s",
            color=PRED_COLOR, ms=4, lw=2.0, label="pred")
    ax.set_xlim(12, -12)
    ax.set_ylim(0, 40)
    ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    return fig


def main():
    """Run the checkpoint over cached samples and write one figure per sample."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="exp/smoke_feature_cache", help="feature cache directory")
    ap.add_argument("--ckpt", default="ckpts/safedrive_phase3_10ep.ckpt")
    ap.add_argument("--config",
                    default="navsim/planning/script/config/common/agent/"
                            "SafeDrive_Phase3_Planner_FullTrain.yaml")
    ap.add_argument("--out", default="exp/viz")
    ap.add_argument("--n", type=int, default=10, help="number of samples")
    ap.add_argument("--score-thr", type=float, default=0.35, help="detection score threshold")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(args.config)
    cfg.checkpoint_path = args.ckpt
    agent = instantiate(cfg)                    # __init__ loads the checkpoint
    agent.eval().to(device)

    sample_dirs = sorted(glob.glob(os.path.join(args.cache, "*", "*", "")))[: args.n]
    print(f"[viz] {len(sample_dirs)} samples, device={device}")
    ades, fdes, mious = [], [], []

    for idx, sample_dir in enumerate(sample_dirs):
        features, targets = custom_collate_fn([load_cached_sample(sample_dir)])
        raw_token = targets["scene_frame_token"]
        token = raw_token[0] if isinstance(raw_token, (list, tuple)) else str(raw_token)
        cam_img = camera_image(features["camera_feature"][0])

        with torch.no_grad():
            pred_raw = agent.forward(to_device(features, device), to_device(targets, device))

        layer = max(int(k.rsplit("_", 1)[1]) for k in pred_raw if k.startswith("ins_states_"))
        tgt = {k: (v[0].cpu().numpy() if torch.is_tensor(v) else v) for k, v in targets.items()}

        gt_boxes, valid = gt_boxes_from_targets(tgt)
        gt = {
            "bev": tgt["bev_semantic_map"].astype(np.int64),
            "boxes": gt_boxes,
            "motion": tgt["motion_traj"][valid][:, 1:, :2],
            "motion_mask": tgt["motion_mask"][valid][:, 1:].astype(bool),
            "plan": tgt["trajectory"],
        }

        score = pred_raw[f"ins_labels_{layer}"][0].sigmoid().cpu().numpy()
        keep = score > args.score_thr
        pred_boxes = pred_boxes_from_states(pred_raw[f"ins_states_{layer}"][0].cpu().numpy())[keep]
        motion_key = f"agent_motion_traj_{layer}"
        motion = None
        if motion_key in pred_raw:
            delta = pred_raw[motion_key][0].cpu().numpy()[keep][:, 0, :, :2]
            motion = delta + pred_boxes[:, None, :2]        # displacement -> ego frame
        pred_plan = pred_raw["trajectory"][0].cpu().numpy()

        pred = {
            "bev": pred_raw["bev_semantic_map"][0].argmax(0).cpu().numpy(),
            "boxes": pred_boxes,
            "motion": motion,
            "plan": pred_plan,
            "iou": per_class_iou(gt["bev"], pred_raw["bev_semantic_map"][0].argmax(0).cpu().numpy()),
            "ade": float(np.linalg.norm(gt["plan"][:, :2] - pred_plan[:, :2], axis=-1).mean()),
            "fde": float(np.linalg.norm(gt["plan"][-1, :2] - pred_plan[-1, :2])),
        }

        fig = plot_sample(cam_img, token, gt, pred, args.score_thr)
        path = os.path.join(args.out, f"{idx:02d}_{token}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)

        ades.append(pred["ade"])
        fdes.append(pred["fde"])
        mious.append(np.nanmean(pred["iou"]))
        print(f"[viz] {path}  GT {len(gt['boxes']):3d} / pred {len(pred_boxes):3d} boxes | "
              f"mIoU {mious[-1]:.3f} | ADE {pred['ade']:.2f} FDE {pred['fde']:.2f}")

    if ades:
        print(f"\n[viz] {len(ades)} samples: mean ADE {np.mean(ades):.3f} m | "
              f"mean FDE {np.mean(fdes):.3f} m | mean mIoU {np.mean(mious):.3f}")

if __name__ == "__main__":
    main()
