"""
mIoU = (mIoU_actionable + mIoU_transformed) / 2 (SPOC paper Eq. 2, Sec. C.3).
GT masks: palettized PNG, 0=bg 1=actionable 2=transformed 4=ambiguous 5=ignore
(4/5 excluded from the metric); predicted masks use 0/1/2.
"""

import argparse
import os
import os.path as osp

import numpy as np
from PIL import Image

ACT, TRF, AMB, IGNORE = 1, 2, 4, 5


def load_mask(path):
    return np.array(Image.open(path).convert("P"), dtype=np.uint8)


def video_iou(gt_dir, pred_dir):
    """Per-video (intersection, union) per class, summed over frames."""
    inter = {ACT: 0, TRF: 0}
    union = {ACT: 0, TRF: 0}
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".png"):
            continue
        gt = load_mask(osp.join(gt_dir, fname))
        valid = (gt != AMB) & (gt != IGNORE)
        if not valid.any():
            continue
        pred_path = osp.join(pred_dir, fname)
        if osp.isfile(pred_path):
            pred = load_mask(pred_path)
            if pred.shape != gt.shape:
                pred = np.array(
                    Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
                )
        else:
            pred = np.zeros_like(gt)  # missing prediction counts as empty
        for c in (ACT, TRF):
            g = (gt == c) & valid
            p = (pred == c) & valid
            inter[c] += int((g & p).sum())
            union[c] += int((g | p).sum())
    return inter, union


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="WTC-HowTo or WTC-VOST")
    p.add_argument("--verb", required=True)
    p.add_argument("--split", default="eval")
    p.add_argument("--pred-dir", required=True, help="Root of predicted masks: <pred-dir>/<osc>/<video_name>/*.png")
    p.add_argument("--noun", default=None)
    p.add_argument("--skip-missing", action="store_true",
                   help="skip videos with no prediction dir (partial eval) instead of scoring them 0")
    args = p.parse_args()

    datadir = osp.join("data/WhereToChange/eval", args.dataset)
    split_file = osp.join("data/WhereToChange/metadata", args.dataset, "subset", args.verb, f"{args.split}.txt")
    with open(split_file) as f:
        videos = [line.strip() for line in f if line.strip()]
    if args.noun:
        videos = [v for v in videos if v.startswith(f"{args.verb}_{args.noun}/")]

    ious = {ACT: [], TRF: []}
    for vid in videos:
        osc, video_name = vid.split("/", 1)
        gt_dir = osp.join(datadir, osc, "gt/masks", video_name)
        if not osp.isdir(gt_dir):
            print(f"WARN: no gt at {gt_dir}, skipping")
            continue
        if args.skip_missing and not osp.isdir(osp.join(args.pred_dir, osc, video_name)):
            continue
        inter, union = video_iou(gt_dir, osp.join(args.pred_dir, osc, video_name))
        parts = []
        for c, name in ((ACT, "act"), (TRF, "trf")):
            if union[c] > 0:
                iou = inter[c] / union[c]
                ious[c].append(iou)
                parts.append(f"{name}={iou:.3f}")
        print(f"{vid}: {' '.join(parts) if parts else 'no gt regions'}")

    miou_act = float(np.mean(ious[ACT])) if ious[ACT] else float("nan")
    miou_trf = float(np.mean(ious[TRF])) if ious[TRF] else float("nan")
    print(f"\n{args.dataset}/{args.verb} ({len(videos)} videos)")
    print(f"mIoU_act = {miou_act:.4f}  (n={len(ious[ACT])})")
    print(f"mIoU_trf = {miou_trf:.4f}  (n={len(ious[TRF])})")
    print(f"mIoU     = {(miou_act + miou_trf) / 2:.4f}")


if __name__ == "__main__":
    main()
