"""Ground-truth laundering patterns, mapped to graph edge ids.

The explanation-fidelity metric needs to know, for a flagged transaction, which other
transactions belong to the *same* laundering scheme (fan-out, cycle, scatter-gather, ...).
Two sources provide that:

  * synthetic  — the generator tags every illicit edge with the motif instance it was drawn
                 from (`_pattern_id`, `_typology` columns); exact, used for CPU testing.
  * AMLworld   — the real HI-Small_Patterns.txt groups transactions into laundering attempts,
                 one typology per attempt; we match each listed transaction back to a graph
                 edge by (bank, account, bank, account, time, amount).

Both resolve to the same `PatternSet`: a per-edge pattern id + typology, aligned to graph edge
order (edges are in the loader's time-sorted row order). Person 1 owns validating the exact
HI-Small_Patterns.txt format against the file on the server; the parser here follows the
published AMLworld layout and is defensive about spacing/case.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.loader import TransactionData

PATTERN_ID_COL = "_pattern_id"
TYPOLOGY_COL = "_typology"


@dataclass
class PatternSet:
    edge_pattern_id: np.ndarray   # [n_edges] pattern instance id, -1 = not in any pattern
    edge_typology: np.ndarray     # [n_edges] typology string ("" when none)
    source: str

    def has_pattern(self, edge_id: int) -> bool:
        return self.edge_pattern_id[edge_id] >= 0

    def members(self, edge_id: int) -> np.ndarray:
        """Edge ids sharing this edge's pattern instance (includes the edge itself)."""
        pid = self.edge_pattern_id[edge_id]
        if pid < 0:
            return np.empty(0, dtype=np.int64)
        return np.where(self.edge_pattern_id == pid)[0]

    def coverage(self) -> float:
        pos = self.edge_pattern_id >= 0
        return float(pos.mean()) if len(pos) else 0.0


def load_patterns(cfg: dict, data: TransactionData) -> PatternSet | None:
    """Pick a provider: synthetic tags if present on the frame, else the AMLworld file."""
    if PATTERN_ID_COL in data.df.columns:
        return from_transaction_data(data)
    path = cfg.get("data", {}).get("patterns_path")
    if path:
        try:
            return from_amlworld_file(path, data, cfg["data"]["columns"])
        except FileNotFoundError:
            return None
    return None


def from_transaction_data(data: TransactionData) -> PatternSet:
    """Synthetic provider: read the generator's motif tags off the (time-sorted) frame."""
    if PATTERN_ID_COL not in data.df.columns:
        raise ValueError(
            f"{PATTERN_ID_COL!r} not on the frame — regenerate synthetic data with pattern "
            "tags (scripts/make_synthetic.py writes them by default)."
        )
    pid = data.df[PATTERN_ID_COL].to_numpy().astype(np.int64)
    typ = data.df[TYPOLOGY_COL].astype(str).to_numpy()
    typ[pid < 0] = ""
    return PatternSet(edge_pattern_id=pid, edge_typology=typ, source="synthetic")


# --------------------------------------------------------------------- AMLworld patterns file

_BEGIN = re.compile(r"BEGIN LAUNDERING ATTEMPT\s*[-:]?\s*(.*)", re.IGNORECASE)
_END = re.compile(r"END LAUNDERING ATTEMPT", re.IGNORECASE)


def from_amlworld_file(path: str, data: TransactionData, col: dict) -> PatternSet:
    """Parse HI-Small_Patterns.txt and align each attempt's transactions to graph edges.

    Matching key = (from_bank, from_account, to_bank, to_account, timestamp, amount_paid),
    the same fields that identify a transaction row in the loader. Duplicate transactions
    (same key) are all assigned the attempt's pattern id.
    """
    attempts = _parse_attempts(path)

    key_to_edges = _edge_key_index(data, col)
    n = len(data.src)
    edge_pid = np.full(n, -1, dtype=np.int64)
    edge_typ = np.array([""] * n, dtype=object)

    for pid, (typology, rows) in enumerate(attempts):
        for row in rows:
            key = _row_key(row, col)
            for e in key_to_edges.get(key, ()):  # unmatched keys are skipped
                if edge_pid[e] < 0:
                    edge_pid[e] = pid
                    edge_typ[e] = typology
    return PatternSet(edge_pattern_id=edge_pid, edge_typology=edge_typ.astype(str), source="amlworld")


def _parse_attempts(path: str) -> list[tuple[str, list[dict]]]:
    """Return [(typology, [transaction-dict, ...]), ...] from the patterns file."""
    header = None  # column names from the real Trans.csv; patterns file lines are headerless
    attempts: list[tuple[str, list[dict]]] = []
    cur_typ, cur_rows = None, []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            mb = _BEGIN.match(line.strip())
            if mb:
                cur_typ, cur_rows = _clean_typology(mb.group(1)), []
                continue
            if _END.match(line.strip()):
                if cur_typ is not None:
                    attempts.append((cur_typ, cur_rows))
                cur_typ, cur_rows = None, []
                continue
            if line.lower().startswith("timestamp,"):  # a repeated header row, if present
                header = line.split(",")
                continue
            if cur_typ is not None:
                parsed = _parse_txn_line(line, header)
                if parsed is not None:
                    cur_rows.append(parsed)
    return attempts


# AMLworld Trans.csv column order (used when a pattern line has no header to key off).
_AML_COLS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]


def _parse_txn_line(line: str, header: list[str] | None) -> dict | None:
    parts = line.split(",")
    cols = header if header is not None else _AML_COLS
    if len(parts) < len(cols):
        return None
    return dict(zip(cols, parts))


def _clean_typology(raw: str) -> str:
    return re.sub(r"[-:]+$", "", raw).strip().upper() or "UNKNOWN"


def _edge_key_index(data: TransactionData, col: dict) -> dict[tuple, list[int]]:
    df = data.df
    ts = pd.to_datetime(df[col["timestamp"]], errors="coerce")
    keys = zip(
        df[col["from_bank"]].astype(str), df[col["from_account"]].astype(str),
        df[col["to_bank"]].astype(str), df[col["to_account"]].astype(str),
        ts.dt.strftime("%Y/%m/%d %H:%M"),
        _amount_str(df[col["amount_paid"]]),
    )
    index: dict[tuple, list[int]] = {}
    for e, key in enumerate(keys):
        index.setdefault(key, []).append(e)
    return index


def _row_key(row: dict, col: dict) -> tuple:
    ts = pd.to_datetime(row.get(col["timestamp"], ""), errors="coerce")
    ts_str = ts.strftime("%Y/%m/%d %H:%M") if pd.notna(ts) else str(row.get(col["timestamp"], ""))
    return (
        str(row.get(col["from_bank"], "")).strip(),
        str(row.get(col["from_account"], "")).strip(),
        str(row.get(col["to_bank"], "")).strip(),
        str(row.get(col["to_account"], "")).strip(),
        ts_str,
        _amount_str(pd.Series([row.get(col["amount_paid"], "")])).iloc[0],
    )


def _amount_str(s: pd.Series) -> pd.Series:
    """Normalize amounts to a stable string key (drops trailing-zero / float-format noise)."""
    num = pd.to_numeric(s, errors="coerce")
    return num.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
