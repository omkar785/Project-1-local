#!/usr/bin/env bash
# Same regime-controlled ablation as run_ablation.sh, but the independent rungs run in two
# lanes — one per GPU — concurrently. This is the right kind of parallelism for an ablation
# (many independent runs); no DDP needed. Runs within a lane are sequential (one model per
# GPU at a time); the two lanes run at once, ~halving wall-clock.
#
#   DATA_CSV=/path/to/HI-Small_Trans.csv DEVICE_A=cuda:1 DEVICE_B=cuda:0 \
#     bash scripts/run_ablation_parallel.sh
#
# Put the FREER GPU in DEVICE_A (it gets the heavier full-graph reference rung). Concurrent
# runs log to the same results/experiments.csv safely (file-locked). Per-lane stdout goes to
# results/lane_A.log / lane_B.log; tail them to watch. Summarize with report_ablation.py.
set -euo pipefail

: "${DATA_CSV:?set DATA_CSV to the HI-Small_Trans.csv path}"
FG_EPOCHS="${FG_EPOCHS:-150}"
MB_EPOCHS="${MB_EPOCHS:-8}"
MB_PATIENCE="${MB_PATIENCE:-4}"
WORKERS="${WORKERS:-4}"
DEVICE_A="${DEVICE_A:-cuda:0}"
DEVICE_B="${DEVICE_B:-cuda:1}"
SEED="${SEED:-42}"
mkdir -p results

common=(--config configs/default.yaml
        --set data.source=csv --set data.csv_path="$DATA_CSV"
        --set experiment.seed="$SEED")
mb="--set train.mode=minibatch --set train.epochs=$MB_EPOCHS \
    --set train.early_stop_patience=$MB_PATIENCE --set train.num_workers=$WORKERS \
    --set model.arch=multi_gin"

# One lane = a device + a list of "name|extra-args" rungs, run sequentially.
run_lane() {
  local device="$1"; shift
  for spec in "$@"; do
    local name="${spec%%|*}" args="${spec#*|}"
    echo ">>> [$device] $name"
    python -m src.train "${common[@]}" --set experiment.device="$device" \
      --set experiment.name="$name" $args
  done
}

# Lane A (freer GPU): full-graph reference + two minibatch arch rungs.
laneA=(
  "gin_fullgraph|--set train.epochs=$FG_EPOCHS --set train.early_stop_patience=25 --set model.arch=gin"
  "gine_mb|$mb --set model.use_reverse_mp=false"
  "gine_reverse_ports|$mb --set model.use_reverse_mp=true --set graph.add_ports=true"
)
# Lane B: GIN-minibatch regime control + the remaining arch rungs.
laneB=(
  "gin_minibatch|$mb --set model.arch=gin"
  "gine_reverse|$mb --set model.use_reverse_mp=true"
  "multigin_full|$mb --set model.use_reverse_mp=true --set graph.add_ports=true --set model.use_ego_ids=true"
)

run_lane "$DEVICE_A" "${laneA[@]}" > results/lane_A.log 2>&1 &
pidA=$!
run_lane "$DEVICE_B" "${laneB[@]}" > results/lane_B.log 2>&1 &
pidB=$!

echo "Lane A ($DEVICE_A) pid $pidA -> results/lane_A.log"
echo "Lane B ($DEVICE_B) pid $pidB -> results/lane_B.log"
echo "Watch:  tail -f results/lane_A.log results/lane_B.log"
wait $pidA; wait $pidB
echo
echo "Both lanes done. Summarize with:  python scripts/report_ablation.py"
