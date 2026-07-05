"""
Diagnostics over stage-1 proposals + label.py outputs.

  grid    threshold grid search over (tau, delta) from stored scores
  phrases compare state-phrase candidates using cached vision embeddings
  oracle  upper-bound mIoU giving every masklet its GT-best label
"""

import argparse
import json
import os
import os.path as osp

import numpy as np
from PIL import Image

from label import apply_constraints, label_from_scores

ACT, TRF = 1, 2
LABEL_VAL = {"act": ACT, "trf": TRF}


# loading
def gt_frames(gt_dir, masks_dir):
    """{stem: (id_mask, gt)} for every GT-annotated frame."""
    frames = {}
    for m in sorted(os.listdir(gt_dir)):
        if not m.endswith(".png"):
            continue
        stem = osp.splitext(m)[0]
        gt = np.array(Image.open(osp.join(gt_dir, m)).convert("P"), dtype=np.uint8)
        mp = osp.join(masks_dir, m)
        idm = (np.array(Image.open(mp).convert("P"), dtype=np.uint8)
               if osp.isfile(mp) else np.zeros_like(gt))
        frames[stem] = (idm, gt)
    return frames


def iter_videos(args, need_labels=True, need_embs=False):
    """Yield (osc, noun, video_name, labels_dir) for the verb's videos with GT."""
    root = args.labels_root if need_labels else args.props_root
    for osc in sorted(os.listdir(root)):
        if not osc.startswith(f"{args.verb}_"):
            continue
        noun = osc.split("_", 1)[1].replace("_", " ")
        sub = osp.join(root, osc, "labels") if need_labels else osp.join(root, osc)
        if not osp.isdir(sub):
            continue
        names = ({osp.splitext(f)[0] for f in os.listdir(sub) if f.endswith(".json")}
                 if need_labels else set(os.listdir(sub)))
        for vn in sorted(names):
            gt_dir = osp.join(args.datadir, osc, "gt/masks", vn)
            if not osp.isdir(gt_dir):
                continue
            if need_embs and not osp.isfile(osp.join(sub, f"{vn}.npz")):
                continue
            yield osc, noun, vn, sub


def video_iu(masklets, frames, tau, delta):
    """Relabel from scores at (tau, delta), refine, return per-class (I, U)."""
    refined = {}
    for mid, fr in masklets.items():
        stems = sorted(fr)
        seq = {i: label_from_scores(fr[s]["s_act"], fr[s]["s_trf"], tau, delta)
               for i, s in enumerate(stems)}
        seq = apply_constraints(seq)
        refined[mid] = {s: seq[i] for i, s in enumerate(stems)}
    inter = {ACT: 0, TRF: 0}
    union = {ACT: 0, TRF: 0}
    for stem, (idm, gt) in frames.items():
        valid = (gt != 4) & (gt != 5)
        pred = np.zeros_like(gt)
        for mid in np.unique(idm):
            if mid == 0:
                continue
            label = refined.get(str(mid), {}).get(stem)
            if label in LABEL_VAL:
                pred[idm == mid] = LABEL_VAL[label]
        for c in (ACT, TRF):
            g = (gt == c) & valid
            p = (pred == c) & valid
            inter[c] += int((g & p).sum())
            union[c] += int((g | p).sum())
    return inter, union


def summarize(ious, tag):
    miou = (np.mean(ious[ACT]) + np.mean(ious[TRF])) / 2
    print(f"{tag}  mIoU={miou:.4f} (act={np.mean(ious[ACT]):.4f} trf={np.mean(ious[TRF]):.4f})")
    return miou


# subcommands
def cmd_grid(args):
    videos = [(json.load(open(osp.join(sub, f"{vn}.json")))["masklets"],
               gt_frames(osp.join(args.datadir, osc, "gt/masks", vn), osp.join(args.props_root, osc, vn)))
              for osc, _, vn, sub in iter_videos(args)]
    print(f"loaded {len(videos)} videos")
    best = None
    for tau in args.taus:
        for delta in args.deltas:
            ious = {ACT: [], TRF: []}
            for masklets, frames in videos:
                inter, union = video_iu(masklets, frames, tau, delta)
                for c in (ACT, TRF):
                    if union[c] > 0:
                        ious[c].append(inter[c] / union[c])
            miou = summarize(ious, f"tau={tau:.2f} delta={delta:.3f}")
            if best is None or miou > best[0]:
                best = (miou, tau, delta)
    print(f"\nbest: mIoU={best[0]:.4f} at tau={best[1]}, delta={best[2]}")


PHRASE_CANDIDATES = {
    "base":     {"act": ["whole {n}"], "trf": ["chopped {n} pieces"]},
    "photo-of": {"act": ["a photo of a whole {n}"], "trf": ["a photo of chopped {n} pieces"]},
    "uncut":    {"act": ["a whole uncut {n}"], "trf": ["small chopped pieces of {n}"]},
    "ensemble": {"act": ["whole {n}", "a whole uncut {n}", "an intact {n}"],
                 "trf": ["chopped {n} pieces", "small chopped pieces of {n}", "diced {n}"]},
}


def cmd_phrases(args):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from label import as_tensor, pick_device
    device = pick_device()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_state(phrases):
        with torch.no_grad():
            t = processor(text=phrases, return_tensors="pt", padding=True).to(device)
            e = as_tensor(model.get_text_features(**t))
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(dim=0)
            return e / e.norm()

    videos = [(noun, dict(np.load(osp.join(sub, f"{vn}.npz"))),
               gt_frames(osp.join(args.datadir, osc, "gt/masks", vn), osp.join(args.props_root, osc, vn)))
              for osc, noun, vn, sub in iter_videos(args, need_embs=True)]
    print(f"loaded {len(videos)} videos")
    nouns = sorted({n for n, _, _ in videos})
    for name, cand in PHRASE_CANDIDATES.items():
        text = {n: torch.stack([encode_state([t.format(n=n) for t in cand["act"]]),
                                encode_state([t.format(n=n) for t in cand["trf"]])]).cpu().numpy().T
                for n in nouns}
        for tau in args.taus:
            ious = {ACT: [], TRF: []}
            for noun, embs, frames in videos:
                masklets = {}
                for key, z in embs.items():
                    stem, mid = key.rsplit("/", 1)
                    s = z @ text[noun]
                    masklets.setdefault(mid, {})[stem] = {"s_act": float(s[0]), "s_trf": float(s[1])}
                inter, union = video_iu(masklets, frames, tau, args.deltas[0])
                for c in (ACT, TRF):
                    if union[c] > 0:
                        ious[c].append(inter[c] / union[c])
            summarize(ious, f"{name:10s} tau={tau:.2f}")


def cmd_oracle(args):
    ious = {ACT: [], TRF: []}
    n_videos = 0
    multi = total = 0
    for osc, _, vn, _ in iter_videos(args, need_labels=False):
        frames = gt_frames(osp.join(args.datadir, osc, "gt/masks", vn), osp.join(args.props_root, osc, vn))
        inter = {ACT: 0, TRF: 0}
        union = {ACT: 0, TRF: 0}
        for idm, gt in frames.values():
            valid = (gt != 4) & (gt != 5)
            pred = np.zeros_like(gt)
            ids = [i for i in np.unique(idm) if i != 0]
            total += 1
            multi += len(ids) >= 2
            for mid in ids:
                sel = (idm == mid) & valid
                if not sel.any():
                    continue
                fr_act = ((gt == ACT) & sel).sum() / sel.sum()
                fr_trf = ((gt == TRF) & sel).sum() / sel.sum()
                best, fr = (ACT, fr_act) if fr_act >= fr_trf else (TRF, fr_trf)
                if fr >= args.min_overlap:
                    pred[sel] = best
            for c in (ACT, TRF):
                g = (gt == c) & valid
                p = (pred == c) & valid
                inter[c] += int((g & p).sum())
                union[c] += int((g | p).sum())
        n_videos += 1
        for c in (ACT, TRF):
            if union[c] > 0:
                ious[c].append(inter[c] / union[c])
    print(f"{n_videos} videos; frames with >=2 masklets: {multi}/{total} ({100 * multi / max(total, 1):.0f}%)")
    summarize(ious, "oracle")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["grid", "phrases", "oracle"])
    p.add_argument("--dataset", default="WTC-HowTo")
    p.add_argument("--verb", required=True)
    p.add_argument("--labels-root", default=None, help="per-OSC dirs containing labels/*.json")
    p.add_argument("--props-root", default=None, help="per-OSC dirs of masklet ID masks")
    p.add_argument("--taus", type=float, nargs="+", default=[0.40, 0.45, 0.50, 0.55, 0.60])
    p.add_argument("--deltas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03, 0.05])
    p.add_argument("--min-overlap", type=float, default=0.5)
    args = p.parse_args()
    args.datadir = f"data/WhereToChange/eval/{args.dataset}"
    args.props_root = args.props_root or f"outputs/proposals/{args.dataset}"
    {"grid": cmd_grid, "phrases": cmd_phrases, "oracle": cmd_oracle}[args.cmd](args)


if __name__ == "__main__":
    main()
