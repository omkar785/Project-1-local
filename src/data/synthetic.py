"""Synthetic AMLworld-HI-Small-shaped transaction generator.

Produces a CSV with the SAME columns as the real HI-Small_Trans.csv so the rest of
the pipeline (loader -> graph -> model) runs identically on the laptop (CPU) and on
the GPU server. It is NOT meant to be realistic AML data — only structurally faithful:
same schema, a directed multigraph of accounts, a small illicit minority, and a few
injected structural motifs (fan-out / cycle) so explainability has something to find.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CURRENCIES = ["US Dollar", "Euro", "Yuan", "Rupee", "Yen"]
PAYMENT_FORMATS = ["ACH", "Cheque", "Credit Card", "Wire", "Cash"]


def generate(
    n_accounts: int = 2000,
    n_transactions: int = 40000,
    illicit_ratio: float = 0.02,
    n_currencies: int = 3,
    n_payment_formats: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    currencies = CURRENCIES[:n_currencies]
    formats = PAYMENT_FORMATS[:n_payment_formats]

    banks = rng.integers(0, 50, size=n_accounts)
    # 8-hex-digit account ids, AMLworld style
    account_ids = np.array([f"{rng.integers(0, 16**8):08X}" for _ in range(n_accounts)])

    n_illicit = int(n_transactions * illicit_ratio)
    n_licit = n_transactions - n_illicit

    rows = []
    base_ts = pd.Timestamp("2022-09-01 00:00:00")

    # --- licit background traffic ---
    for _ in range(n_licit):
        s, r = rng.integers(0, n_accounts, size=2)
        while r == s:
            r = rng.integers(0, n_accounts)
        rows.append(_row(rng, base_ts, banks, account_ids, s, r, currencies, formats, label=0))

    # --- illicit motifs: fan-out and cycle, so structure carries signal ---
    remaining = n_illicit
    while remaining > 0:
        motif = rng.choice(["fan_out", "cycle"])
        if motif == "fan_out":
            k = min(rng.integers(3, 8), remaining)
            src = rng.integers(0, n_accounts)
            dsts = rng.choice(n_accounts, size=int(k), replace=False)
            for d in dsts:
                if d == src:
                    continue
                rows.append(_row(rng, base_ts, banks, account_ids, src, d, currencies, formats, label=1))
            remaining -= int(k)
        else:  # cycle
            k = min(rng.integers(3, 6), remaining)
            chain = rng.choice(n_accounts, size=int(k), replace=False)
            for i in range(len(chain)):
                s, r = chain[i], chain[(i + 1) % len(chain)]
                rows.append(_row(rng, base_ts, banks, account_ids, s, r, currencies, formats, label=1))
            remaining -= int(k)

    df = pd.DataFrame(rows)
    # shuffle then sort by time so the temporal split is meaningful
    df = df.sample(frac=1.0, random_state=seed).sort_values("Timestamp").reset_index(drop=True)
    return df


def _row(rng, base_ts, banks, account_ids, s, r, currencies, formats, label):
    minutes = int(rng.integers(0, 60 * 24 * 30))  # spread over ~30 days
    ts = (base_ts + pd.Timedelta(minutes=minutes)).strftime("%Y/%m/%d %H:%M")
    # Mild, realistic edge-feature signal so the baseline is learnable on synthetic data:
    # illicit transactions skew to larger amounts and cross-currency payments. Kept mild
    # (overlapping distributions) so the graph structure still has to do work.
    if label == 1:
        amt = float(np.round(rng.lognormal(mean=6.8, sigma=1.2), 2))
        pay_cur, recv_cur = _maybe_cross(rng, currencies, p_cross=0.6)
    else:
        amt = float(np.round(rng.lognormal(mean=6.0, sigma=1.2), 2))
        pay_cur, recv_cur = _maybe_cross(rng, currencies, p_cross=0.1)
    return {
        "Timestamp": ts,
        "From Bank": int(banks[s]),
        "Account": account_ids[s],
        "To Bank": int(banks[r]),
        "Account.1": account_ids[r],
        "Amount Received": amt,
        "Receiving Currency": recv_cur,
        "Amount Paid": amt,
        "Payment Currency": pay_cur,
        "Payment Format": rng.choice(formats),
        "Is Laundering": int(label),
    }


def _maybe_cross(rng, currencies, p_cross):
    """Return (payment_currency, receiving_currency), cross-currency with prob p_cross."""
    pay = rng.choice(currencies)
    if len(currencies) > 1 and rng.random() < p_cross:
        recv = rng.choice([c for c in currencies if c != pay])
    else:
        recv = pay
    return pay, recv
