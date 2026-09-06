"""Write a synthetic AMLworld-shaped CSV (same schema as HI-Small_Trans.csv).

    python scripts/make_synthetic.py --out data/synthetic_trans.csv --n-transactions 40000

Lets you eyeball the schema and run the `data.source=csv` path locally without the
real dataset.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic_trans.csv")
    p.add_argument("--n-accounts", type=int, default=2000)
    p.add_argument("--n-transactions", type=int, default=40000)
    p.add_argument("--illicit-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = generate(n_accounts=args.n_accounts, n_transactions=args.n_transactions,
                  illicit_ratio=args.illicit_ratio, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")
    print(f"illicit: {int(df['Is Laundering'].sum())} "
          f"({100*df['Is Laundering'].mean():.2f}%)")
    print("columns:", list(df.columns))


if __name__ == "__main__":
    main()
