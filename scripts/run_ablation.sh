#!/usr/bin/env bash
# Week-2 Multi-GIN ablation ladder (Person 2 deliverable).
# Runs the four variants from the plan and appends each to results/experiments.csv.
#
#   DATA_CSV=/path/to/HI-Small_Trans.csv EPOCHS=150 bash scripts/run_ablation.sh
#
# Then inspect the table with:  python scripts/report_ablation.py
set -euo pipefail

: "${DATA_CSV:?set DATA_CSV to the HI-Small_Trans.csv path}"
EPOCHS="${EPOCHS:-150}"
DEVICE="${DEVICE:-cuda:0}"
PATIENCE="${PATIENCE:-25}"
SEED="${SEED:-42}"

common=(--config configs/default.yaml
        --set data.source=csv
        --set data.csv_path="$DATA_CSV"
        --set experiment.device="$DEVICE"
        --set experiment.seed="$SEED"
        --set train.epochs="$EPOCHS"
        --set train.early_stop_patience="$PATIENCE")

echo "=== 1/4  plain GIN (baseline) ==="
python -m src.train "${common[@]}" \
  --set model.arch=gin \
  --set experiment.name=gin_baseline

echo "=== 2/4  Multi-GIN: + reverse MP ==="
python -m src.train "${common[@]}" \
  --set model.arch=multi_gin --set model.use_reverse_mp=true --set graph.add_ports=false \
  --set experiment.name=multigin_reverse

echo "=== 3/4  Multi-GIN: + reverse MP + ports ==="
python -m src.train "${common[@]}" \
  --set model.arch=multi_gin --set model.use_reverse_mp=true --set graph.add_ports=true \
  --set experiment.name=multigin_reverse_ports

echo "=== 4/4  Multi-GIN (edge-aware, no reverse) — isolates the reverse-MP contribution ==="
python -m src.train "${common[@]}" \
  --set model.arch=multi_gin --set model.use_reverse_mp=false --set graph.add_ports=true \
  --set experiment.name=multigin_ports_only

echo
echo "Done. Summarize with:  python scripts/report_ablation.py"
