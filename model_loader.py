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


def load_prediction_backend() -> Tuple[Kind | None, Backend | None]:
    """
    MODEL_BACKEND=auto|bundle|pkl
    auto: prefer artifacts/model_bundle.joblib if present, else xg_boost_best_model.pkl + meta.
    """
    mode = os.environ.get("MODEL_BACKEND", "auto").lower()
    bundle_path = Path(os.environ.get("MODEL_PATH", _root() / "artifacts/model_bundle.joblib"))
    pkl_path = Path(os.environ.get("PKL_MODEL_PATH", _root() / "xg_boost_best_model.pkl"))
    meta_path = Path(os.environ.get("PKL_META_PATH", _root() / "xg_boost_best_model.meta.json"))

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
