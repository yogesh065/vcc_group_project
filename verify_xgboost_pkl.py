#!/usr/bin/env python3
"""Check xgboost_model.pkl vs xgboost_model.meta.json (needs working xgboost, e.g. Linux or macOS + libomp)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib


def main() -> None:
    root = Path(__file__).resolve().parent
    pkl = root / "xgboost_model.pkl"
    meta_path = root / "xgboost_model.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    m = joblib.load(pkl)
    nf = int(getattr(m, "n_features_in_", -1))
    n_meta = len(meta["feature_names"])
    print("Loaded:", type(m).__name__)
    print("n_features_in_:", nf, "| meta feature_names:", n_meta, "| OK:", nf == n_meta)
    print("classes_:", getattr(m, "classes_", None))


if __name__ == "__main__":
    main()
