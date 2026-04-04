#!/usr/bin/env python3
"""Train DDoS detector and save joblib bundle for Streamlit / AWS inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ddos_pipeline import fit_bundle_from_raw

ROOT = Path(__file__).resolve().parent
DEFAULT_CICIDS_CSV = ROOT / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DDoS model (CICIDS-style CSV).")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_CICIDS_CSV,
        help=f"Labeled flow CSV (default: {DEFAULT_CICIDS_CSV.name}). Column names are str-stripped; label column ' Label' → 'Label'.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/model_bundle.joblib"),
        help="Output path for the saved model bundle.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast training (smaller XGBoost, no grid search).",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run GridSearchCV like the notebook (slow).",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Data file not found: {args.data}")

    df = pd.read_csv(args.data)
    bundle = fit_bundle_from_raw(df, use_grid_search=args.grid, quick_train=args.quick)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"Saved bundle → {args.out}")
    print("Hold-out metrics:", bundle.meta)


if __name__ == "__main__":
    main()
