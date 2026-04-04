"""
Inference for a plain sklearn estimator saved as .pkl/.joblib (e.g. RandomForest).
Uses `xg_boost_best_model.meta.json`: `feature_names` (order) and optional `class_names_by_index`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import joblib
import numpy as np
import pandas as pd

from ddos_pipeline import LABEL_COL


class SklearnPklBackend:
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        class_display_names: List[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        n = int(getattr(model, "n_features_in_", len(feature_names)))
        if len(feature_names) != n:
            raise ValueError(
                f"meta.json has {len(feature_names)} features but model expects {n}"
            )
        classes = list(getattr(model, "classes_", []))
        if len(class_display_names) != len(classes):
            raise ValueError(
                f"class_names length {len(class_display_names)} != model.classes_ {len(classes)}"
            )
        self._class_to_display = {c: lab for c, lab in zip(classes, class_display_names)}

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        work = df.copy()
        work.columns = work.columns.str.strip()
        if LABEL_COL in work.columns:
            work = work.drop(columns=[LABEL_COL])
        for c in self.feature_names:
            if c not in work.columns:
                work[c] = np.nan
        sub = work[self.feature_names]
        sub = sub.apply(pd.to_numeric, errors="coerce")
        sub.replace([np.inf, -np.inf], np.nan, inplace=True)
        med = sub.median(numeric_only=True)
        sub = sub.fillna(med).fillna(0.0)
        return sub.to_numpy(dtype=np.float64)

    def predict_all(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self._build_X(df)
        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)
        labels = np.array([self._class_to_display[p] for p in pred])
        return pred, proba, labels


def load_pkl_backend(pkl_path: Path, meta_path: Path) -> SklearnPklBackend:
    model = joblib.load(pkl_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = meta["feature_names"]
    classes = list(getattr(model, "classes_", []))
    by_idx: List[str] | None = meta.get("class_names_by_index")
    if by_idx is not None:
        display = [by_idx[int(c)] for c in classes]
    else:
        display = [str(c) for c in classes]
    return SklearnPklBackend(model, names, display)
