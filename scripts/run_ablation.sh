#!/usr/bin/env bash
# Week-2 Multi-GIN ablation ladder (Person 2 deliverable). Five rungs so the ego-ID gain
# is cleanly separated from the change in training regime:
#   1 plain GIN                              (full-graph)
#   2 + reverse MP                           (full-graph)
#   3 + reverse MP + ports                   (full-graph)
#   4 + reverse MP + ports, minibatch NO-ego (control for the sampling regime)
#   5 + reverse MP + ports + ego IDs         (minibatch)  <- full Multi-GIN
# Each run appends a row to results/experiments.csv.
#
#   DATA_CSV=/path/to/HI-Small_Trans.csv bash scripts/run_ablation.sh
#
# Rungs 4-5 need mini-batch sampling -> require pyg-lib (see SERVER.md).
# Summarize with report_ablation.py.
set -euo pipefail

: "${DATA_CSV:?set DATA_CSV to the HI-Small_Trans.csv path}"
EPOCHS="${EPOCHS:-150}"          # full-graph rungs (~1.3s/epoch)
MB_EPOCHS="${MB_EPOCHS:-6}"      # minibatch rungs (~750 grad steps/epoch -> converges fast)
MB_PATIENCE="${MB_PATIENCE:-3}"
WORKERS="${WORKERS:-4}"          # parallel samplers; set 0 if you hit a /dev/shm bus error
DEVICE="${DEVICE:-cuda:0}"
PATIENCE="${PATIENCE:-25}"
SEED="${SEED:-42}"

common=(--config configs/default.yaml
        --set data.source=csv --set data.csv_path="$DATA_CSV"
        --set experiment.device="$DEVICE" --set experiment.seed="$SEED")
fg=(--set train.epochs="$EPOCHS" --set train.early_stop_patience="$PATIENCE")
mb=(--set train.mode=minibatch --set train.epochs="$MB_EPOCHS"
    --set train.early_stop_patience="$MB_PATIENCE" --set train.num_workers="$WORKERS"
    --set model.arch=multi_gin --set model.use_reverse_mp=true --set graph.add_ports=true)

echo "=== 1/5  plain GIN (baseline) ==="
python -m src.train "${common[@]}" "${fg[@]}" \
  --set model.arch=gin --set experiment.name=gin_baseline

echo "=== 2/5  Multi-GIN: + reverse MP ==="
python -m src.train "${common[@]}" "${fg[@]}" \
  --set model.arch=multi_gin --set model.use_reverse_mp=true \
  --set experiment.name=multigin_reverse

echo "=== 3/5  Multi-GIN: + reverse MP + ports ==="
python -m src.train "${common[@]}" "${fg[@]}" \
  --set model.arch=multi_gin --set model.use_reverse_mp=true --set graph.add_ports=true \
  --set experiment.name=multigin_reverse_ports

echo "=== 4/5  Multi-GIN minibatch, NO ego (regime control) ==="
python -m src.train "${common[@]}" "${mb[@]}" \
  --set model.use_ego_ids=false --set experiment.name=multigin_mb_noego

echo "=== 5/5  full Multi-GIN: + ego IDs (minibatch) ==="
python -m src.train "${common[@]}" "${mb[@]}" \
  --set model.use_ego_ids=true --set experiment.name=multigin_full

echo
echo "Done. Summarize with:  python scripts/report_ablation.py"
