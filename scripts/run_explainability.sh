#!/usr/bin/env bash
# Week-2 Layer-6 driver: explain the detector's alerts and score explanation fidelity
# (the paper's Claim 3, ">=80% of true positives"). Runs on a SAVED checkpoint — train the
# keeper first with experiment.save_checkpoint=true.
#
# Produces:
#   results/<TAG>_fidelity.json   aggregate + per-typology fidelity
#   results/<TAG>_records.json    per-alert explanation records for the demo
#
#   # local CPU sanity (synthetic, pattern tags built in):
#   CKPT=results/checkpoints/keeper_sparse_seed42.pt bash scripts/run_explainability.sh
#
#   # server, real data + patterns file:
#   CKPT=results/checkpoints/keeper_seed42.pt DEVICE=cuda \
#   CSV=/path/HI-Small_Trans.csv PATTERNS=/path/HI-Small_Patterns.txt \
#   bash scripts/run_explainability.sh
#
# Fidelity is CPU-cheap (one small subgraph per alert); no pyg-lib needed.
set -euo pipefail

: "${CKPT:?set CKPT to a keeper checkpoint (.pt) from train.py}"
DEVICE="${DEVICE:-cpu}"
NUM_EXPLAIN="${NUM_EXPLAIN:-500}"
NUM_RECORDS="${NUM_RECORDS:-20}"
TAG="${TAG:-explain}"
OUT_DIR="${OUT_DIR:-results}"

overrides=(--set "experiment.device=${DEVICE}")
# CSV/PATTERNS override the checkpoint's data location for the real dataset on the server.
[ -n "${CSV:-}" ] && overrides+=(--set data.source=csv --set "data.csv_path=${CSV}")
[ -n "${PATTERNS:-}" ] && overrides+=(--set "data.patterns_path=${PATTERNS}")
[ -n "${NROWS:-}" ] && overrides+=(--set "data.nrows=${NROWS}")

python scripts/run_explainability.py \
  --checkpoint "$CKPT" \
  --num-explain "$NUM_EXPLAIN" --num-records "$NUM_RECORDS" \
  --device "$DEVICE" --out-dir "$OUT_DIR" --tag "$TAG" \
  "${overrides[@]}"

echo
echo "Done. Fidelity: ${OUT_DIR}/${TAG}_fidelity.json   Demo records: ${OUT_DIR}/${TAG}_records.json"
