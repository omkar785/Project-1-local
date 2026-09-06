"""Load transactions (real CSV or synthetic), engineer features, temporal split.

Output is a `TransactionData` bundle consumed by src/graph/build.py. Keeping the
DataFrame -> tensor boundary here means the graph/model code never has to know about
raw column names (Person 1 owns this file; Person 2 consumes the bundle).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TransactionData:
    df: pd.DataFrame                 # cleaned, sorted by time, with integer account ids
    src: np.ndarray                  # sender node index per transaction
    dst: np.ndarray                  # receiver node index per transaction
    edge_features: np.ndarray        # [n_tx, n_feat] transaction features
    labels: np.ndarray               # [n_tx] Is Laundering
    timestamps: np.ndarray           # [n_tx] unix seconds
    split: np.ndarray                # [n_tx] 0=train 1=val 2=test
    n_nodes: int
    account_ids: np.ndarray          # [n_nodes] original "bank_account" string per node index
    feature_names: list[str]


def load(cfg: dict) -> TransactionData:
    dcfg = cfg["data"]
    if dcfg["source"] == "synthetic":
        from src.data.synthetic import generate

        s = dcfg["synthetic"]
        df = generate(
            n_accounts=s["n_accounts"],
            n_transactions=s["n_transactions"],
            illicit_ratio=s["illicit_ratio"],
            n_currencies=s["n_currencies"],
            n_payment_formats=s["n_payment_formats"],
            seed=cfg["experiment"]["seed"],
        )
    elif dcfg["source"] == "csv":
        df = pd.read_csv(dcfg["csv_path"])
    else:
        raise ValueError(f"Unknown data.source: {dcfg['source']}")

    col = dcfg["columns"]
    return _build(df, col, dcfg["split"])


def _build(df: pd.DataFrame, col: dict, split_cfg: dict) -> TransactionData:
    # --- timestamps ---
    ts = pd.to_datetime(df[col["timestamp"]], errors="coerce")
    df = df.assign(_ts=ts).sort_values("_ts").reset_index(drop=True)
    unix = (df["_ts"].astype("int64") // 10**9).to_numpy()

    # --- account node index (bank + account uniquely identifies a party) ---
    src_key = df[col["from_bank"]].astype(str) + "_" + df[col["from_account"]].astype(str)
    dst_key = df[col["to_bank"]].astype(str) + "_" + df[col["to_account"]].astype(str)
    uniq, inv = np.unique(np.concatenate([src_key.to_numpy(), dst_key.to_numpy()]), return_inverse=True)
    n = len(src_key)
    src = inv[:n]
    dst = inv[n:]

    # --- edge (transaction) features: label-free, numeric ---
    feats, names = _edge_features(df, col)

    labels = df[col["label"]].astype(int).to_numpy()
    split = _temporal_split(unix, split_cfg) if split_cfg["method"] == "temporal" else _random_split(n, split_cfg)

    return TransactionData(
        df=df,
        src=src.astype(np.int64),
        dst=dst.astype(np.int64),
        edge_features=feats.astype(np.float32),
        labels=labels.astype(np.int64),
        timestamps=unix,
        split=split,
        n_nodes=len(uniq),
        account_ids=uniq,
        feature_names=names,
    )


def _edge_features(df: pd.DataFrame, col: dict) -> tuple[np.ndarray, list[str]]:
    parts, names = [], []

    amount = pd.to_numeric(df[col["amount_paid"]], errors="coerce").fillna(0.0).to_numpy()
    parts.append(np.log1p(amount).reshape(-1, 1)); names.append("log_amount_paid")

    amount_r = pd.to_numeric(df[col["amount_received"]], errors="coerce").fillna(0.0).to_numpy()
    parts.append(np.log1p(amount_r).reshape(-1, 1)); names.append("log_amount_received")

    # cross-currency flag (a known laundering signal)
    cross = (df[col["payment_currency"]].astype(str) != df[col["receiving_currency"]].astype(str)).astype(float)
    parts.append(cross.to_numpy().reshape(-1, 1)); names.append("cross_currency")

    # cyclical time-of-day / day-of-week
    ts = df["_ts"]
    hour = ts.dt.hour.fillna(0).to_numpy()
    dow = ts.dt.dayofweek.fillna(0).to_numpy()
    parts.append(np.sin(2 * np.pi * hour / 24).reshape(-1, 1)); names.append("hour_sin")
    parts.append(np.cos(2 * np.pi * hour / 24).reshape(-1, 1)); names.append("hour_cos")
    parts.append(np.sin(2 * np.pi * dow / 7).reshape(-1, 1)); names.append("dow_sin")
    parts.append(np.cos(2 * np.pi * dow / 7).reshape(-1, 1)); names.append("dow_cos")

    # one-hot payment format + currency (low cardinality)
    for field, prefix in ((col["payment_format"], "fmt"), (col["payment_currency"], "cur")):
        dummies = pd.get_dummies(df[field].astype(str), prefix=prefix)
        parts.append(dummies.to_numpy(dtype=float)); names.extend(list(dummies.columns))

    return np.concatenate(parts, axis=1), names


def _temporal_split(unix: np.ndarray, cfg: dict) -> np.ndarray:
    # rows already sorted by time; split by cumulative fraction
    n = len(unix)
    tr = int(n * cfg["train_frac"])
    va = int(n * (cfg["train_frac"] + cfg["val_frac"]))
    split = np.full(n, 2, dtype=np.int64)
    split[:tr] = 0
    split[tr:va] = 1
    return split


def _random_split(n: int, cfg: dict) -> np.ndarray:
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    tr = int(n * cfg["train_frac"])
    va = int(n * (cfg["train_frac"] + cfg["val_frac"]))
    split = np.full(n, 2, dtype=np.int64)
    split[idx[:tr]] = 0
    split[idx[tr:va]] = 1
    return split
