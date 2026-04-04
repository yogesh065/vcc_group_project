"""
DDoS detection pipeline aligned with DDoS_Detection_ProDS_MTech(2).ipynb:
preprocess → variance/correlation pruning → MI top-30 → RobustScaler →
optional SMOTE (training only) → classifier.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")

ID_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
LABEL_COL = "Label"
CORR_THRESHOLD = 0.97
TOP_N = 30
MI_SAMPLE_CAP = 50_000
RNG = np.random.default_rng(42)


@dataclass
class ModelBundle:
    """Serializable artifact for training / inference."""

    medians: pd.Series
    zero_var_cols: List[str]
    high_corr_cols: List[str]
    selected_features: List[str]
    scaler: RobustScaler
    label_encoder: LabelEncoder
    classifier: Any
    meta: Dict[str, Any] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        Xp = preprocess_for_predict(df, self)
        proba = self.classifier.predict_proba(Xp)
        pred = self.classifier.predict(Xp)
        return pred, proba


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()
    return out


def build_xy_from_raw(
    df_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, pd.Series]:
    df = _strip_columns(df_raw)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected column '{LABEL_COL}' in dataset.")

    drop_cols = [c for c in ID_COLS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    medians = X.median(numeric_only=True)
    X.fillna(medians, inplace=True)

    dup_mask = df.duplicated()
    if dup_mask.any():
        df = df.loc[~dup_mask].copy()
        X = X.loc[df.index].copy()
        y = y.loc[df.index]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le, medians


def prune_features(
    X: pd.DataFrame, y_enc: np.ndarray
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    zero_var = X.columns[X.var() == 0].tolist()
    X = X.drop(columns=zero_var, errors="ignore")

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
    X = X.drop(columns=high_corr, errors="ignore")

    n = min(MI_SAMPLE_CAP, len(X))
    sample_idx = RNG.choice(len(X), size=n, replace=False)
    mi_scores = mutual_info_classif(X.iloc[sample_idx], y_enc[sample_idx], random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    selected = mi_series.head(TOP_N).index.tolist()
    X_sel = X[selected]
    return X_sel, zero_var, high_corr, selected


def imbalance_ratio(y_enc: np.ndarray) -> float:
    counts = np.bincount(y_enc)
    return counts.max() / max(counts.min(), 1)


def preprocess_for_predict(df: pd.DataFrame, bundle: ModelBundle) -> np.ndarray:
    df = _strip_columns(df)
    drop_cols = [c for c in ID_COLS if c in df.columns]
    work = df.drop(columns=drop_cols + [LABEL_COL], errors="ignore")

    work = work.apply(pd.to_numeric, errors="coerce")
    work.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in bundle.medians.index:
        if col not in work.columns:
            work[col] = np.nan
    work = work.reindex(columns=list(bundle.medians.index), fill_value=np.nan)
    work.fillna(bundle.medians, inplace=True)

    work = work.drop(columns=bundle.zero_var_cols, errors="ignore")
    work = work.drop(columns=bundle.high_corr_cols, errors="ignore")

    missing = [c for c in bundle.selected_features if c not in work.columns]
    if missing:
        raise ValueError(f"Input missing required features: {missing[:5]}...")

    Xs = work[bundle.selected_features]
    return bundle.scaler.transform(Xs)


def fit_bundle_from_raw(
    df_raw: pd.DataFrame,
    use_grid_search: bool = False,
    quick_train: bool = False,
) -> ModelBundle:
    """
    Train end-to-end. `quick_train` uses a small fixed XGBoost for fast demos.
    `use_grid_search` matches the notebook (slow).
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import GridSearchCV, train_test_split
    from xgboost import XGBClassifier

    X, y_enc, le, medians = build_xy_from_raw(df_raw)
    X_sel, zero_var, high_corr, selected = prune_features(X, y_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    ir = imbalance_ratio(y_enc)
    if ir > 1.5:
        smote = SMOTE(random_state=42)
        X_train_sc, y_train = smote.fit_resample(X_train_sc, y_train)

    xgb_n_jobs = int(os.environ.get("XGBOOST_N_JOBS", "1"))

    if quick_train:
        clf = XGBClassifier(
            n_estimators=80,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            random_state=42,
            eval_metric="logloss",
            n_jobs=xgb_n_jobs,
        )
        clf.fit(X_train_sc, y_train)
    elif use_grid_search:
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
        gs = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=xgb_n_jobs),
            grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=int(os.environ.get("GRIDSEARCH_N_JOBS", "1")),
            verbose=1,
        )
        gs.fit(X_train_sc, y_train)
        clf = gs.best_estimator_
    else:
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
            eval_metric="logloss",
            n_jobs=xgb_n_jobs,
        )
        clf.fit(X_train_sc, y_train)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    pred = clf.predict(X_test_sc)
    prob = clf.predict_proba(X_test_sc)
    pos = 1 if prob.shape[1] > 1 else 0
    try:
        roc = float(roc_auc_score(y_test, prob[:, pos]))
    except ValueError:
        roc = float("nan")
    meta = {
        "imbalance_ratio": float(ir),
        "smote_applied": ir > 1.5,
        "holdout_accuracy": float(accuracy_score(y_test, pred)),
        "holdout_f1_macro": float(f1_score(y_test, pred, average="macro")),
        "holdout_roc_auc": roc,
    }

    return ModelBundle(
        medians=medians,
        zero_var_cols=zero_var,
        high_corr_cols=high_corr,
        selected_features=selected,
        scaler=scaler,
        label_encoder=le,
        classifier=clf,
        meta=meta,
    )
