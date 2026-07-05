"""
Stage 1 on Modal: mask proposal generation with GroundingDINO + SAM + DeAOT
(via SAM-Track), per SPOC paper Sec. 3.2 / Table C hyperparameters.

Per clip: extract frames at 5 fps, detect the noun with GroundingDINO every 10
frames, segment with SAM, track with DeAOT in between. Saves masklet-ID masks
(palettized PNG, pixel = track ID) at 1 fps, named to match JPEGImages_1fps.
"""

import os
import subprocess

import modal

app = modal.App("spoc-proposals")
vol = modal.Volume.from_name("spoc-data", create_if_missing=True)
VOL = "/vol"

DATA_URL = (
    "https://utexas.app.box.com/index.php?rm=box_download_shared_file"
    "&shared_name=tx52j9gcgq7s89mq8ldwkwzcn5jukbwe&file_id=f_2142926206992"
)
SAMTRACK = "/root/samtrack"

# Table C hyperparameters
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.5
BOX_SIZE_THRESHOLD = 0.7
SAM_GAP = 10        # detect every 10 frames (2s at 5fps)
MIN_AREA = 50
MAX_OBJ_NUM = 255
MIN_NEW_OBJ_IOU = 0.8
FPS = 5

image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "wget", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2", "torchvision==0.16.2",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "wheel", "setuptools",
        "opencv-python", "pillow", "numpy<2", "gdown", "scipy",
        "transformers==4.38.2", "addict", "yapf", "timm", "pycocotools",
        "supervision==0.6.0", "matplotlib", "scikit-image", "imageio",
    )
    .env({"CC": "gcc", "CXX": "g++"})  # standalone python defaults to clang, absent here
    .run_commands(
        f"git clone --depth 1 https://github.com/z-x-yang/Segment-and-Track-Anything.git {SAMTRACK}",
        f"pip install -e {SAMTRACK}/sam",
        "pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git",
        # pre-download GroundingDINO's BERT text encoder into the image
        "python -c \"from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')\"",
        # checkpoints (Table C: SAM vit_h, DeAOT r50_deaotl PRE_YTB_DAV, GroundingDINO swint)
        f"mkdir -p {SAMTRACK}/ckpt",
        f"wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O {SAMTRACK}/ckpt/sam_vit_h_4b8939.pth",
        f"wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O {SAMTRACK}/ckpt/groundingdino_swint_ogc.pth",
        f"gdown 1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ -O {SAMTRACK}/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
        gpu="t4",  # GroundingDINO builds its CUDA ops when a GPU is visible
    )
    .run_commands(
        # tool/detector.py reads its config from a source checkout at this path
        f"git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git {SAMTRACK}/groundingdino",
    )
    .env({"PYTHONPATH": f"{SAMTRACK}:{SAMTRACK}/aot"})
)


@app.function(image=modal.Image.debian_slim().apt_install("curl", "unzip"), volumes={VOL: vol}, timeout=3600 * 4)
def download_dataset():
    """Download the WhereToChange eval zip from Box straight into the volume and unzip."""
    zip_path = f"{VOL}/WhereToChange.zip"
    if not os.path.exists(f"{VOL}/data/WhereToChange"):
        subprocess.run(["curl", "-sL", "-C", "-", "-o", zip_path, DATA_URL], check=True)
        os.makedirs(f"{VOL}/data", exist_ok=True)
        subprocess.run(["unzip", "-q", zip_path, "-d", f"{VOL}/data"], check=True)
        os.remove(zip_path)
        vol.commit()
    subprocess.run(["find", f"{VOL}/data", "-maxdepth", "3", "-type", "d"], check=True)


@app.function(image=image, gpu="A10G", volumes={VOL: vol}, timeout=1800)
def extract_proposals(dataset: str, osc: str, video_name: str):
    """Run detect+segment+track on one clip; save 1fps masklet-ID masks to the volume."""
    import glob
    import shutil

    import cv2
    import numpy as np
    from PIL import Image as PILImage

    os.chdir(SAMTRACK)
    from model_args import aot_args, sam_args, segtracker_args
    from SegTracker import SegTracker

    # v2 = point-grid intra-object proposals (C.1); v1 (one mask per box) at outputs/proposals
    out_dir = f"{VOL}/outputs/proposals_v2/{dataset}/{osc}/{video_name}"
    if os.path.isdir(out_dir) and any(f.endswith(".png") for f in os.listdir(out_dir)):
        return f"{osc}/{video_name}: already done, skipping"

    noun = osc.split("_", 1)[1].replace("_", " ")
    clip_dir = f"{VOL}/data/WhereToChange/eval/{dataset}/{osc}/clips"
    clips = glob.glob(f"{clip_dir}/{video_name}.*")
    assert clips, f"no clip for {clip_dir}/{video_name}"
    clip_path = clips[0]

    # 1 fps frame stems encode native-fps frame indices (e.g. frame00024 at 24fps
    # = t=1s), and the grid is not always anchored at t=0. Map each stem to its
    # timestamp, then to the nearest extracted 5fps frame index.
    jpeg_dir = f"{VOL}/data/WhereToChange/eval/{dataset}/{osc}/JPEGImages_1fps/{video_name}"
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(jpeg_dir) if f.endswith((".jpg", ".jpeg")))
    fps_str = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", clip_path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    num, den = fps_str.split("/")
    native_fps = float(num) / float(den)

    frames_dir = "/tmp/frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", clip_path, "-vf", f"fps={FPS}",
         "-start_number", "0", f"{frames_dir}/%06d.jpg"],
        check=True,
    )
    frame_paths = sorted(glob.glob(f"{frames_dir}/*.jpg"))

    idx_to_stems = {}
    for stem in stems:
        t = int(stem.replace("frame", "")) / native_fps
        idx = min(round(t * FPS), len(frame_paths) - 1)
        idx_to_stems.setdefault(idx, []).append(stem)

    sam_args["sam_checkpoint"] = f"{SAMTRACK}/ckpt/sam_vit_h_4b8939.pth"
    sam_args["model_type"] = "vit_h"
    # paper C.1: 32x32 point grid, NMS 0.9 -> multiple intra-object masks per box
    sam_args["generator_args"] = {
        "points_per_side": 32,
        "box_nms_thresh": 0.9,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.9,
        "crop_n_layers": 0,
        "min_mask_region_area": MIN_AREA,
    }
    aot_args["model"] = "r50_deaotl"
    aot_args["model_path"] = f"{SAMTRACK}/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth"
    aot_args["long_term_mem_gap"] = 9999
    aot_args["max_len_long_term"] = 9999
    segtracker_args["sam_gap"] = SAM_GAP
    segtracker_args["min_area"] = MIN_AREA
    segtracker_args["max_obj_num"] = MAX_OBJ_NUM
    segtracker_args["min_new_obj_iou"] = MIN_NEW_OBJ_IOU

    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()

    os.makedirs(out_dir, exist_ok=True)
    palette = [0, 0, 0] + [((i * 63) % 256) for i in range(3 * 255)]

    import torch

    def detect_grid(frame):
        """GroundingDINO boxes gate SAM 32x32 point-grid masks (paper C.1)."""
        h, w = frame.shape[:2]
        _, boxes = segtracker.detector.run_grounding(frame, noun, BOX_THRESHOLD, TEXT_THRESHOLD)
        boxes = [b for b in boxes
                 if (b[1][0] - b[0][0]) * (b[1][1] - b[0][1]) <= h * w * BOX_SIZE_THRESHOLD]
        merged = np.zeros((h, w), dtype=np.uint8)
        if not boxes:
            return merged
        anns = segtracker.sam.everything_generator.generate(frame)
        idx = 0
        for ann in sorted(anns, key=lambda a: -a["area"]):  # finer masks paint over coarser
            m = ann["segmentation"]
            ys, xs = np.nonzero(m)
            if len(ys) < MIN_AREA or idx >= MAX_OBJ_NUM:
                continue
            in_box = max(((xs >= b[0][0]) & (xs <= b[1][0]) &
                          (ys >= b[0][1]) & (ys <= b[1][1])).mean() for b in boxes)
            if in_box >= 0.7:
                idx += 1
                merged[m] = idx
        return merged

    initialized = False
    max_id = 0
    with torch.cuda.amp.autocast():
        for idx, fp in enumerate(frame_paths):
            frame = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            if not initialized:
                # re-detect every frame until the object first appears
                pred_mask = detect_grid(frame)
                if pred_mask.any():
                    segtracker.add_reference(frame, pred_mask)
                    initialized = True
            elif idx % SAM_GAP == 0:
                seg_mask = detect_grid(frame)
                track_mask = segtracker.track(frame)
                segtracker.curr_idx = max_id  # find_new_objs numbers new ids from here
                new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
                pred_mask = track_mask + new_obj_mask
                segtracker.add_reference(frame, pred_mask)
            else:
                pred_mask = segtracker.track(frame, update_memory=True)
            max_id = max(max_id, int(pred_mask.max()))
            torch.cuda.empty_cache()

            for stem in idx_to_stems.get(idx, []):
                im = PILImage.fromarray(pred_mask.astype(np.uint8), mode="P")
                im.putpalette(palette)
                im.save(f"{out_dir}/{stem}.png")

    vol.commit()
    return f"{osc}/{video_name}: {len(frame_paths)} frames -> {out_dir}"


@app.local_entrypoint()
def main(verb: str, dataset: str = "WTC-HowTo", noun: str = None, video_name: str = None, limit: int = 0):
    split_file = f"data/WhereToChange/metadata/{dataset}/subset/{verb}/eval.txt"
    with open(split_file) as f:
        videos = [line.strip() for line in f if line.strip()]
    if noun:
        videos = [v for v in videos if v.startswith(f"{verb}_{noun}/")]
    if video_name:
        videos = [v for v in videos if v.endswith(f"/{video_name}")]
    if limit:
        videos = videos[:limit]
    print(f"{len(videos)} clips")
    args = [(dataset, *v.split("/", 1)) for v in videos]
    for result in extract_proposals.starmap(args, return_exceptions=True):
        print(result)
