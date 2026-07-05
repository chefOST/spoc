#!/bin/zsh
# SPOC (PL) for one verb: CLIP label + constraints + paint + eval.
# Stage-1 proposals must already be local (modal volume get spoc-data outputs/proposals_v2 outputs/).
# Usage: ./run_spoc_pl.sh chopping [WTC-HowTo] [outputs/proposals_v2]
set -e
VERB=${1:?verb required}
DATASET=${2:-WTC-HowTo}
PROPS=${3:-outputs/proposals_v2}/$DATASET
OUT=outputs/pl/$DATASET
PY=.venv/bin/python

mkdir -p "${OUT}_masks"
for OSC_DIR in $PROPS/${VERB}_*; do
  [ -d "$OSC_DIR" ] || continue
  OSC=$(basename "$OSC_DIR")
  $PY pseudolabel/label.py --osc "$OSC" \
    --frames-dir "data/WhereToChange/eval/$DATASET/$OSC/JPEGImages_1fps" \
    --masks-dir "$OSC_DIR" \
    --out-dir "$OUT/$OSC"
  ln -sfn "$(pwd)/$OUT/$OSC/masks" "${OUT}_masks/$OSC"  # eval friendly layout
done

$PY eval/evaluate_miou.py --dataset "$DATASET" --verb "$VERB" --pred-dir "${OUT}_masks" --skip-missing
