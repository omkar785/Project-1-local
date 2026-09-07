# SentinelGraph

Label-efficient anti-money-laundering (AML) detection on transaction graphs, combining
**self-supervised representation learning** (Stage 1) with a **directed-multigraph GNN
detector** (Stage 2, Multi-GIN). Target dataset: **AMLworld HI-Small**.

This repo is the working pipeline for the 4-week / 50%-milestone plan. It is developed and
smoke-tested locally on CPU with a **synthetic AMLworld-shaped generator**, then run for
real on the GPU server (see [SERVER.md](SERVER.md)).

## Problem framing

Transaction-level AML detection as **edge classification**:

```
nodes  = accounts (bank + account id)
edges  = transactions (directed multigraph — parallel edges kept)
label  = "Is Laundering" per transaction
```

Metrics are for **severe class imbalance**: primary = minority-class **F1** and **PR-AUC**
(accuracy is not reported as headline).

## Layout

```
configs/default.yaml     all knobs; override per run with --set key=value
src/
  config.py              yaml + dotted-key CLI overrides
  data/
    synthetic.py         AMLworld-shaped synthetic generator (same schema as real CSV)
    loader.py            CSV/synthetic -> cleaned df, features, temporal split  [Person 1]
  graph/build.py         directed-multigraph PyG Data + edge split masks        [Person 1]
  models/gin.py          baseline GIN edge classifier                           [Person 2]
  metrics.py             minority-F1, PR-AUC, precision/recall, confusion       [Person 2]
  train.py               train/eval CLI, class weighting, label-% subsampling   [Person 2]
  utils/                 seeding, experiment CSV logging
scripts/make_synthetic.py   materialize a synthetic CSV to inspect the schema
results/experiments.csv     append-only log (seed, label%, metrics, time)
```

## Quickstart (local, CPU, synthetic data)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torch_geometric        # CPU is fine locally
pip install -r requirements.txt

# smoke test the whole pipeline on synthetic data
python -m src.train --config configs/default.yaml --set train.epochs=5

# a label-efficiency point (10% of train laundering labels)
python -m src.train --config configs/default.yaml --label-pct 10
```

Running on the real data + GPU: see **[SERVER.md](SERVER.md)**.

## Status vs. the 4-week plan

| Plan item | Status |
|---|---|
| Data preprocessing + temporal split + graph | ✅ real-schema loader, config-driven |
| Baseline GIN + metrics + weighted loss | ✅ verified on real HI-Small |
| **Directed Multi-GIN**: edge-aware (GINE) + reverse MP + ports | ✅ full-graph, ablatable |
| **Full Multi-GIN**: + ego IDs (mini-batch LinkNeighborLoader) | ✅ code done, GPU run pending |
| Week-2 ablation ladder (`scripts/run_ablation.sh`) | ✅ one-command GIN→full Multi-GIN table |
| Label-efficiency scaffolding (`--label-pct`) | ✅ works in both train modes |
| Experiment logging (seed/label%/arch/metrics/time) | ✅ |
| **Stage 1 SSL** (LaundroGraph-style) | ⏳ Person 3 (Week 1–2) |
| Stage 1 → Stage 2 integration | ⏳ Week 3 |
| Ablations + explainability prototype | ⏳ Week 4 |

Deferred (per plan §8): AMLworld Medium/Large, Elliptic, dashboard, federated/DP/drift, SAR.

Owner focus: this repo is driven for **Person 2 (GNN & detection)**; the data/graph lane is
built to a stable interface so Person 1's work slots in, and Person 3's SSL/integration
attaches at the documented Stage-1 → Stage-2 boundary.
