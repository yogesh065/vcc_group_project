"""Resolve which model artifact to use: joblib pipeline bundle or sklearn pickle."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Tuple

import joblib

from pkl_backend import SklearnPklBackend, load_pkl_backend

Backend = Any  # ModelBundle | SklearnPklBackend
Kind = Literal["bundle", "pkl"]


def _root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_pkl_paths() -> tuple[Path, Path]:
    """PKL_MODEL_PATH / PKL_META_PATH env, else xgboost_model.pkl, else xg_boost_best_model.pkl."""
    root = _root()
    if os.environ.get("PKL_MODEL_PATH"):
        pkl = Path(os.environ["PKL_MODEL_PATH"])
        meta = Path(os.environ.get("PKL_META_PATH", pkl.with_name(pkl.stem + ".meta.json")))
        return pkl, meta
    xgb = root / "xgboost_model.pkl"
    xgb_meta = root / "xgboost_model.meta.json"
    if xgb.is_file() and xgb_meta.is_file():
        return xgb, xgb_meta
    return root / "xg_boost_best_model.pkl", root / "xg_boost_best_model.meta.json"


def load_prediction_backend() -> Tuple[Kind | None, Backend | None]:
    """
    MODEL_BACKEND=auto|bundle|pkl
    auto: prefer model_bundle.joblib if present, else pickle (+ meta).
    Default pickle: xgboost_model.pkl when present, else xg_boost_best_model.pkl.
    """
    mode = os.environ.get("MODEL_BACKEND", "auto").lower()
    bundle_path = Path(os.environ.get("MODEL_PATH", _root() / "artifacts/model_bundle.joblib"))
    pkl_path, meta_path = _resolve_pkl_paths()

    if mode == "bundle":
        if bundle_path.is_file():
            return "bundle", joblib.load(bundle_path)
        return None, None

    if mode == "pkl":
        if pkl_path.is_file() and meta_path.is_file():
            return "pkl", load_pkl_backend(pkl_path, meta_path)
        return None, None

    # auto
    if bundle_path.is_file():
        return "bundle", joblib.load(bundle_path)
    if pkl_path.is_file() and meta_path.is_file():
        return "pkl", load_pkl_backend(pkl_path, meta_path)
    return None, None
