Implementation of [SPOC](https://arxiv.org/abs/2503.11953) (Mandikal et al., UT Austin) pseudo-labelling, including OSC dynamics constraints, for the WhereToChange benchmark. Labels object regions per frame as **actionable** (1, not yet changed) or **transformed** (2, changed).

**Note**: no training/implementation of the SPOC video transformer model ((c) from Figure 2 below)
![Figure 2](overview.png)

**All credit for the SPOC method and the WhereToChange benchmark belongs to the original authors:**

> Priyanka Mandikal, Tushar Nagarajan, Alex Stoken, Zihui Xue, Kristen Grauman. **SPOC: Spatially-Progressing Object State Change Segmentation in Video**. arXiv:2503.11953, 2025.

[Paper](https://arxiv.org/abs/2503.11953) · [Dataset](https://github.com/priyankamandikal/spoc) · [Project page](https://vision.cs.utexas.edu/projects/spoc-spatially-progressing-osc/)

## Pipeline

- `pseudolabel/proposals_modal.py` — stage 1 (Modal GPU): GroundingDINO + SAM + DeAOT -> masklet-ID masks per frame.
- `pseudolabel/label.py` — stages 2-3 (local): CLIP state scoring -> dynamics constraints -> painted masks.
- `pseudolabel/diagnose.py` — threshold grid / phrase / oracle diagnostics.
- `eval/evaluate_miou.py` — the paper's mIoU metric.
- `run_spoc_pl.sh` — driver: label + paint every OSC of a verb, then eval.

## Usage

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. proposals (Modal GPU; data lives in the spoc-data volume)
modal run pseudolabel/proposals_modal.py::download_dataset   # once, populates the volume
modal volume get spoc-data data data/ --force                # pull WhereToChange locally (needed below)
modal run pseudolabel/proposals_modal.py --verb chopping --noun avocado

# 2-3 + eval (local; fetch proposals from the volume first)
modal volume get spoc-data outputs/proposals_v2 outputs/ --force
./run_spoc_pl.sh chopping

# diagnostics (threshold grid, phrase candidates, proposal oracle ceiling)
python pseudolabel/diagnose.py grid    --verb chopping --labels-root outputs/pl/WTC-HowTo
python pseudolabel/diagnose.py phrases --verb chopping --labels-root outputs/pl/WTC-HowTo
python pseudolabel/diagnose.py oracle  --verb chopping --props-root outputs/proposals_v2/WTC-HowTo
```

## Results (chopping, WTC-HowTo, 157 clips)

| Setup | mIoU |
|---|---|
| SPOC (PL), defaults | 0.277 |
| oracle (GT-best label per masklet) | 0.503 |
| paper: SPOC (PL) / oracle | 0.44-0.46 / 0.696 |

Frame / predicted mask / ground truth, red = actionable, green = transformed:

![chopping avocado example](samples/chopping_avocado.png)
![chopping garlic example](samples/chopping_garlic.png)

