#!/usr/bin/env bash
# Week-2 Multi-GIN ablation ladder (Person 2 deliverable) — the four rungs from the plan:
#   plain GIN  ->  + reverse MP  ->  + reverse MP + ports  ->  full Multi-GIN (+ ego IDs)
# Each run appends a row to results/experiments.csv.
#
#   DATA_CSV=/path/to/HI-Small_Trans.csv bash scripts/run_ablation.sh
#
# Rungs 1-3 use full-graph training (fast on the H100). Rung 4 (ego IDs) needs mini-batch
# subgraph sampling -> requires pyg-lib (see SERVER.md). Summarize with report_ablation.py.
set -euo pipefail

: "${DATA_CSV:?set DATA_CSV to the HI-Small_Trans.csv path}"
EPOCHS="${EPOCHS:-150}"          # full-graph rungs
MB_EPOCHS="${MB_EPOCHS:-25}"     # minibatch rung (many gradient steps/epoch -> fewer epochs)
DEVICE="${DEVICE:-cuda:0}"
PATIENCE="${PATIENCE:-25}"
SEED="${SEED:-42}"

common=(--config configs/default.yaml
        --set data.source=csv --set data.csv_path="$DATA_CSV"
        --set experiment.device="$DEVICE" --set experiment.seed="$SEED"
        --set train.early_stop_patience="$PATIENCE")

echo "=== 1/4  plain GIN (baseline) ==="
python -m src.train "${common[@]}" --set train.epochs="$EPOCHS" \
  --set model.arch=gin --set experiment.name=gin_baseline

echo "=== 2/4  Multi-GIN: + reverse MP ==="
python -m src.train "${common[@]}" --set train.epochs="$EPOCHS" \
  --set model.arch=multi_gin --set model.use_reverse_mp=true \
  --set experiment.name=multigin_reverse

echo "=== 3/4  Multi-GIN: + reverse MP + ports ==="
python -m src.train "${common[@]}" --set train.epochs="$EPOCHS" \
  --set model.arch=multi_gin --set model.use_reverse_mp=true --set graph.add_ports=true \
  --set experiment.name=multigin_reverse_ports

echo "=== 4/4  full Multi-GIN: + ego IDs  (minibatch; may be the slow rung) ==="
python -m src.train "${common[@]}" --set train.epochs="$MB_EPOCHS" \
  --set train.mode=minibatch \
  --set model.arch=multi_gin --set model.use_reverse_mp=true \
  --set graph.add_ports=true --set model.use_ego_ids=true \
  --set experiment.name=multigin_full

echo
echo "Done. Summarize with:  python scripts/report_ablation.py"
