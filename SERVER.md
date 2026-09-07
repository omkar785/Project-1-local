# Running SentinelGraph on the GPU server (manual)

You run this manually on the shared server — the assistant never opens a session there.
Everything is developed and smoke-tested locally on CPU with synthetic data first, so by
the time it reaches the server the only new variables are the **real dataset** and the **GPU**.

> Fill in the two `TODO` values below once and you're set.
>
> - `DATA_CSV` = absolute path to the real `HI-Small_Trans.csv` on the server → **TODO**
> - `PY` = the python you'll use (conda env / venv / module) → **TODO**

## 1. One-time environment setup

Pick whichever the server supports. **torch + torch_geometric must match the server CUDA**,
so install them separately from the CUDA-independent deps.

```bash
# clone / copy this repo onto the server, then from its root:

# (a) create an isolated env  — conda:
conda create -y -n sentinelgraph python=3.11 && conda activate sentinelgraph
#     ...or venv:
# python3.11 -m venv .venv && source .venv/bin/activate

# (b) install torch matching the server CUDA (check with `nvidia-smi`).
#     Example for CUDA 12.1 — adjust the index-url to your CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# (c) install PyG (matches the torch you just installed):
pip install torch_geometric

# (d) install pyg-lib — REQUIRED for minibatch mode (ego IDs / full Multi-GIN).
#     Use your torch + CUDA versions. torch 2.5.x + CUDA 12.1 example:
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# (e) the rest:
pip install -r requirements.txt
```

Verify pyg-lib imports (only needed for the minibatch / ego-ID rung):

```bash
python -c "import pyg_lib; print('pyg_lib OK')"
```

Sanity check the GPU is visible to torch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Point the config at the real data

Confirm the column names match your CSV first:

```bash
head -3 "$DATA_CSV"          # compare against data.columns in configs/default.yaml
```

If any header differs, edit `data.columns` in `configs/default.yaml` (or override per run
with `--set data.columns.label="Is Laundering"` etc.).

## 3. Run

```bash
python -m src.train \
  --config configs/default.yaml \
  --set data.source=csv \
  --set data.csv_path="$DATA_CSV" \
  --set experiment.device=cuda \
  --set experiment.name=gin_baseline_server
```

You'll see per-epoch validation metrics, a final TEST line, wall-clock time, and a row
appended to `results/experiments.csv`. That timing tells you whether the full dataset is
tractable before we scale up.

## 4. Quick timing probe (optional)

To gauge speed without a full run, cap epochs and (if needed) subsample:

```bash
python -m src.train --config configs/default.yaml \
  --set data.source=csv --set data.csv_path="$DATA_CSV" \
  --set experiment.device=cuda --set train.epochs=3
```

## What I need from you to finalize this file
- `nvidia-smi` output (GPU model + VRAM + CUDA version) → sets the torch index-url and later batch sizes.
- `head -3 "$DATA_CSV"` and `ls` of the dataset directory → confirms the schema + the patterns filename.
- Whether the server has internet for `pip` (if air-gapped, we pre-stage wheels instead).
- Python version available on the server (3.10/3.11 recommended; PyG wheels are easiest there).
