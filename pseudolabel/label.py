"""
Pseudo-labeling, stages 2-3 (paper Sec. 3.2-3.3): CLIP state scoring (Eq. 1)
-> dynamics constraints -> painted masks. One pass per OSC.

Inputs:  <frames-dir>/<video>/*.jpg, <masks-dir>/<video>/*.png (masklet-ID masks, stage 1)
Outputs: <out-dir>/labels/<video>.{json,npz}, <out-dir>/masks/<video>/*.png (0=bg 1=act 2=trf)

Self-tests: python pseudolabel/label.py --self-test
"""

import argparse
import json
import os
import os.path as osp

import numpy as np

TAU, DELTA = 0.5, 0.01
ACT, TRF, AMB, BG = "act", "trf", "amb", "bg"
LABEL_VAL = {ACT: 1, TRF: 2}
PALETTE = [0, 0, 0, 255, 102, 102, 153, 255, 153]  # bg black, act red, trf green

# state phrases
# verb -> (actionable template, transformed template); {n} = noun.
# Paper generates these with LLM; add per-OSC OVERRIDES as needed.
VERB_TEMPLATES = {
    "chopping":  ("whole {n}", "chopped {n} pieces"),
    "slicing":   ("whole {n}", "sliced {n} pieces"),
    "mincing":   ("whole {n}", "minced {n}"),
    "grating":   ("whole {n}", "grated {n}"),
    "shredding": ("whole {n}", "shredded {n}"),
    "mashing":   ("whole {n}", "mashed {n}"),
    "crushing":  ("whole {n}", "crushed {n}"),
    "peeling":   ("unpeeled {n}", "peeled {n}"),
    "melting":   ("solid {n}", "melted {n}"),
    "coating":   ("plain uncoated {n}", "{n} coated with topping"),
}
OVERRIDES = {}


def state_phrases(osc):
    """'chopping_avocado' -> ('whole avocado', 'chopped avocado pieces')"""
    if osc in OVERRIDES:
        return OVERRIDES[osc]
    verb, noun = osc.split("_", 1)
    act_t, trf_t = VERB_TEMPLATES[verb]
    noun = noun.replace("_", " ")
    return act_t.format(n=noun), trf_t.format(n=noun)


# CLIP scoring
def label_from_scores(s_act, s_trf, tau=TAU, delta=DELTA):
    """Eq. 1: background / ambiguous / actionable / transformed."""
    if s_act + s_trf < tau:
        return BG
    if abs(s_act - s_trf) < delta:
        return AMB
    return ACT if s_act > s_trf else TRF


def as_tensor(features):
    """transformers <5 returns a tensor; v5 returns an output object."""
    import torch
    return features if torch.is_tensor(features) else features.pooler_output


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bbox_crop(img, mask, mask_out=False):
    from PIL import Image
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    if mask_out:
        arr = np.array(img)
        arr[~mask] = 127  # grey out pixels outside the mask
        img = Image.fromarray(arr)
    return img.crop(box)


# dynamics constraints (3.3)
def causal_ordering(labels):
    """labels: {frame_index: label}. Flip outliers until all act precede trf."""
    labels = dict(labels)
    while True:
        s_act = sorted(t for t, l in labels.items() if l == ACT)
        s_trf = sorted(t for t, l in labels.items() if l == TRF)
        if not s_act or not s_trf or s_act[-1] < s_trf[0]:
            return labels
        mid_act = sum(s_act) / len(s_act)
        mid_trf = sum(s_trf) / len(s_trf)
        if abs(s_act[-1] - mid_act) > abs(s_trf[0] - mid_trf):
            labels[s_act[-1]] = TRF  # last actionable is the outlier
        else:
            labels[s_trf[0]] = ACT  # first transformed is the outlier


def ambiguity_resolution(labels):
    """Snap each 'amb' to act/trf by proximity to the act/trf boundary."""
    labels = dict(labels)
    s_act = [t for t, l in labels.items() if l == ACT]
    s_trf = [t for t, l in labels.items() if l == TRF]
    for t, l in labels.items():
        if l != AMB:
            continue
        if s_act and s_trf:
            labels[t] = ACT if abs(t - max(s_act)) < abs(t - min(s_trf)) else TRF
        elif s_act:
            labels[t] = ACT
        elif s_trf:
            labels[t] = TRF
    return labels


def apply_constraints(labels):
    return ambiguity_resolution(causal_ordering(labels))


# pipeline
def score_video(video_name, args, model, processor, text_emb, device):
    """CLIP-score every (masklet, frame). Returns (masklets dict, embeddings)."""
    import torch
    from PIL import Image
    masks_dir = osp.join(args.masks_dir, video_name)
    frames_dir = osp.join(args.frames_dir, video_name)
    masklets, embs = {}, {}
    for fname in sorted(os.listdir(masks_dir)):
        if not fname.endswith(".png"):
            continue
        stem = osp.splitext(fname)[0]
        frame_path = next((osp.join(frames_dir, stem + e) for e in (".jpg", ".jpeg", ".png")
                           if osp.isfile(osp.join(frames_dir, stem + e))), None)
        if frame_path is None:
            continue
        id_mask = np.array(Image.open(osp.join(masks_dir, fname)).convert("P"))
        img = Image.open(frame_path).convert("RGB")
        crops, ids = [], []
        for mid in np.unique(id_mask):
            if mid == 0:
                continue
            crop = bbox_crop(img, id_mask == mid, mask_out=args.mask_out)
            if crop is not None:
                crops.append(crop)
                ids.append(int(mid))
        if not crops:
            continue
        with torch.no_grad():
            inputs = processor(images=crops, return_tensors="pt").to(device)
            v = as_tensor(model.get_image_features(**inputs))
            v = v / v.norm(dim=-1, keepdim=True)
            scores = (v @ text_emb.T).cpu().numpy()
        for i, (mid, (s_act, s_trf)) in enumerate(zip(ids, scores)):
            masklets.setdefault(str(mid), {})[stem] = {
                "label": label_from_scores(float(s_act), float(s_trf), args.tau, args.delta),
                "s_act": round(float(s_act), 4),
                "s_trf": round(float(s_trf), 4),
            }
            embs[f"{stem}/{mid}"] = v[i].cpu().numpy()
    return masklets, embs


def refine(masklets, skip_constraints=False):
    """{mid: {stem: {label,...}}} -> {mid: {stem: label}} after constraints."""
    out = {}
    for mid, frames in masklets.items():
        stems = sorted(frames)
        seq = {i: frames[s]["label"] for i, s in enumerate(stems)}
        if not skip_constraints:
            seq = apply_constraints(seq)
        out[mid] = {s: seq[i] for i, s in enumerate(stems)}
    return out


def paint_video(video_name, refined, masks_dir, out_dir):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(masks_dir)):
        if not fname.endswith(".png"):
            continue
        stem = osp.splitext(fname)[0]
        id_mask = np.array(Image.open(osp.join(masks_dir, fname)).convert("P"))
        out = np.zeros_like(id_mask, dtype=np.uint8)
        for mid in np.unique(id_mask):
            if mid == 0:
                continue
            label = refined.get(str(mid), {}).get(stem)
            if label in LABEL_VAL:
                out[id_mask == mid] = LABEL_VAL[label]
        im = Image.fromarray(out, mode="P")
        im.putpalette(PALETTE)
        im.save(osp.join(out_dir, fname))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--osc", help="e.g. chopping_avocado")
    p.add_argument("--frames-dir")
    p.add_argument("--masks-dir", help="masklet ID masks per video (stage 1 output)")
    p.add_argument("--out-dir", help="writes labels/ and masks/ under here")
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--delta", type=float, default=DELTA)
    p.add_argument("--video-name", default=None)
    p.add_argument("--mask-out", action="store_true", help="grey out background in crops")
    p.add_argument("--skip-constraints", action="store_true", help="raw CLIP labels ablation")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    for a in ("osc", "frames_dir", "masks_dir", "out_dir"):
        assert getattr(args, a), f"--{a.replace('_', '-')} required"

    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = pick_device()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    act_phrase, trf_phrase = state_phrases(args.osc)
    with torch.no_grad():
        t = processor(text=[act_phrase, trf_phrase], return_tensors="pt", padding=True).to(device)
        text_emb = as_tensor(model.get_text_features(**t))
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    labels_dir = osp.join(args.out_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    videos = [args.video_name] if args.video_name else sorted(os.listdir(args.masks_dir))
    for video_name in videos:
        if not osp.isdir(osp.join(args.masks_dir, video_name)):
            continue
        masklets, embs = score_video(video_name, args, model, processor, text_emb, device)
        with open(osp.join(labels_dir, f"{video_name}.json"), "w") as f:
            json.dump({"osc": args.osc, "phrases": [act_phrase, trf_phrase], "masklets": masklets}, f)
        np.savez_compressed(osp.join(labels_dir, f"{video_name}.npz"), **embs)
        paint_video(video_name, refine(masklets, args.skip_constraints),
                    osp.join(args.masks_dir, video_name), osp.join(args.out_dir, "masks", video_name))
        print(f"{video_name}: {len(masklets)} masklets labeled + painted")


def self_test():
    def seq(s):  # "a a t" -> {0: act, 1: act, 2: trf}
        m = {"a": ACT, "t": TRF, "m": AMB, "b": BG}
        return {i: m[c] for i, c in enumerate(s.split())}

    assert causal_ordering(seq("a a t t")) == seq("a a t t")
    assert causal_ordering(seq("a a t t a")) == seq("a a t t t")
    assert causal_ordering(seq("t a a a t t")) == seq("a a a a t t")
    assert causal_ordering(seq("a b t a"))[1] == BG
    assert causal_ordering(seq("a a a")) == seq("a a a")
    assert ambiguity_resolution(seq("m a t"))[0] == ACT
    assert ambiguity_resolution(seq("a t m"))[2] == TRF
    assert ambiguity_resolution(seq("a a m t"))[2] == TRF  # equidistant tie -> trf
    assert ambiguity_resolution(seq("m b m")) == seq("m b m")
    assert apply_constraints(seq("t a m a t t")) == seq("a a a a t t")
    assert label_from_scores(0.2, 0.2) == BG
    assert label_from_scores(0.3, 0.3) == AMB
    assert label_from_scores(0.35, 0.25) == ACT
    assert label_from_scores(0.25, 0.35) == TRF
    print("all self-tests passed")


if __name__ == "__main__":
    main()
