"""
Inference for sklearn-compatible estimators in .pkl/.joblib (RandomForest, XGBClassifier, etc.).
Sidecar JSON: `feature_names` (training order) and optional `class_names_by_index` for int classes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple

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
        scaler: Any | None = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler
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
        X = sub.to_numpy(dtype=np.float64)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def predict_all(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self._build_X(df)
        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)
        labels = np.array([self._class_to_display[p] for p in pred])
        return pred, proba, labels


def _class_display_names(classes: Sequence[Any], by_idx: List[str] | None) -> List[str]:
    """Map model.classes_ to human labels (handles int 0/1 or str BENIGN/DDoS)."""
    out: List[str] = []
    for c in classes:
        if by_idx is not None and isinstance(c, (int, np.integer)):
            out.append(by_idx[int(c)])
        else:
            out.append(str(c))
    return out


def load_pkl_backend(pkl_path: Path, meta_path: Path) -> SklearnPklBackend:
    model = joblib.load(pkl_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = meta["feature_names"]
    classes = list(getattr(model, "classes_", []))
    by_idx: List[str] | None = meta.get("class_names_by_index")
    display = _class_display_names(classes, by_idx)
    scaler = None
    rel = meta.get("scaler_joblib")
    if rel:
        sp = Path(rel)
        if not sp.is_file():
            sp = meta_path.parent / rel
        if sp.is_file():
            scaler = joblib.load(sp)
    return SklearnPklBackend(model, names, display, scaler=scaler)
