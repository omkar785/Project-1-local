#!/usr/bin/env bash
# Week-3 label-efficiency curve (project's central experiment). Trains the best config
# (GINE + reverse MP + ports, minibatch — no ego, per the Week-2 ablation) at several
# label fractions. The FULL graph structure is always used; only the supervised label
# fraction is thinned (src/train.py --label-pct), so this measures how detection degrades
# as labelled transactions get scarce.
#
# This is the FROM-SCRATCH arm. The Stage-1-pretrained arm is the same command plus
#   --set graph.node_features=embedding --set graph.embedding.path=<stage1.pt>
# once Person 3's embeddings exist; compare the two curves for the label-efficiency claim.
#
#   DATA_CSV=/path/to/HI-Small_Trans.csv DEVICE=cuda:0 bash scripts/run_label_efficiency.sh
#   # multi-seed (Week-4 robustness):  SEEDS="42 1 2" ...
#
# Needs pyg-lib (minibatch). Summarize with scripts/report_label_efficiency.py.
set -euo pipefail

: "${DATA_CSV:?set DATA_CSV to the HI-Small_Trans.csv path}"
DEVICE="${DEVICE:-cuda:0}"
WORKERS="${WORKERS:-4}"
EPOCHS="${MB_EPOCHS:-10}"
PATIENCE="${MB_PATIENCE:-5}"
SEEDS="${SEEDS:-42}"
LABEL_PCTS="${LABEL_PCTS:-1 10 50 100}"

for seed in $SEEDS; do
  for pct in $LABEL_PCTS; do
    echo "=== label% ${pct}  seed ${seed} ==="
    python -m src.train --config configs/default.yaml \
      --set data.source=csv --set data.csv_path="$DATA_CSV" \
      --set experiment.device="$DEVICE" --set experiment.seed="$seed" \
      --set train.mode=minibatch --set train.num_workers="$WORKERS" \
      --set train.epochs="$EPOCHS" --set train.early_stop_patience="$PATIENCE" \
      --set model.arch=multi_gin --set model.use_reverse_mp=true --set graph.add_ports=true \
      --set experiment.name="labeleff_p${pct}_s${seed}" \
      --label-pct "$pct"
  done
done

echo
echo "Done. Curve:  python scripts/report_label_efficiency.py"
