#!/usr/bin/env bash
# Week-2 Multi-GIN ablation ladder (Person 2). REGIME-CONTROLLED: every architecture rung
# runs under the SAME mini-batch training regime, so differences are attributable to the
# architecture and not to full-graph-vs-minibatch optimization. One reference full-graph
# GIN row is included to show the (large) regime effect on its own.
#
#   Rung 0  GIN, full-graph            reference (weak 1-step/epoch optimizer)
#   Rung 1  GIN, minibatch             regime control: same arch, proper SGD
#   Rung 2  GINE, minibatch            + edge-aware message passing
#   Rung 3  GINE + reverse MP          + reverse message passing
#   Rung 4  GINE + reverse + ports     + port numbering
#   Rung 5  GINE + reverse + ports + ego IDs   full Multi-GIN
#
#   DATA_CSV=/path/to/HI-Small_Trans.csv DEVICE=cuda:1 bash scripts/run_ablation.sh
#
# Mini-batch rungs need pyg-lib (see SERVER.md). Summarize with report_ablation.py.
set -euo pipefail

: "${DATA_CSV:?set DATA_CSV to the HI-Small_Trans.csv path}"
FG_EPOCHS="${FG_EPOCHS:-150}"
MB_EPOCHS="${MB_EPOCHS:-8}"
MB_PATIENCE="${MB_PATIENCE:-4}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"

common=(--config configs/default.yaml
        --set data.source=csv --set data.csv_path="$DATA_CSV"
        --set experiment.device="$DEVICE" --set experiment.seed="$SEED")
mb=(--set train.mode=minibatch --set train.epochs="$MB_EPOCHS"
    --set train.early_stop_patience="$MB_PATIENCE" --set train.num_workers="$WORKERS")

echo "=== 0  GIN, full-graph (reference) ==="
python -m src.train "${common[@]}" --set train.epochs="$FG_EPOCHS" --set train.early_stop_patience=25 \
  --set model.arch=gin --set experiment.name=gin_fullgraph

echo "=== 1  GIN, minibatch (regime control) ==="
python -m src.train "${common[@]}" "${mb[@]}" \
  --set model.arch=gin --set experiment.name=gin_minibatch

echo "=== 2  GINE, minibatch (edge-aware MP) ==="
python -m src.train "${common[@]}" "${mb[@]}" \
  --set model.arch=multi_gin --set model.use_reverse_mp=false \
  --set experiment.name=gine_mb

echo "=== 3  GINE + reverse MP ==="
python -m src.train "${common[@]}" "${mb[@]}" \
  --set model.arch=multi_gin --set model.use_reverse_mp=true \
  --set experiment.name=gine_reverse

echo "=== 4  GINE + reverse + ports ==="
python -m src.train "${common[@]}" "${mb[@]}" \
  --set model.arch=multi_gin --set model.use_reverse_mp=true --set graph.add_ports=true \
  --set experiment.name=gine_reverse_ports

echo "=== 5  GINE + reverse + ports + ego IDs (full Multi-GIN) ==="
python -m src.train "${common[@]}" "${mb[@]}" \
  --set model.arch=multi_gin --set model.use_reverse_mp=true --set graph.add_ports=true \
  --set model.use_ego_ids=true --set experiment.name=multigin_full

echo
echo "Done. Summarize with:  python scripts/report_ablation.py"
