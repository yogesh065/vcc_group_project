"""
Streamlit UI for DDoS traffic detection (CICIDS 2017–style flow features).
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ddos_pipeline import LABEL_COL, ModelBundle, fit_bundle_from_raw, preprocess_for_predict
from model_loader import load_prediction_backend
from pkl_backend import SklearnPklBackend

ARTIFACT = Path(os.environ.get("MODEL_PATH", "artifacts/model_bundle.joblib"))
DEFAULT_CSV = Path(__file__).resolve().parent / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"


@st.cache_resource
def get_backend():
    return load_prediction_backend()


def main() -> None:
    st.set_page_config(
        page_title="DDoS Detection",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("🛡️ DDoS traffic detection")
    st.caption(
        "CICIDS-style flow CSV (e.g. Friday-WorkingHours-Afternoon-DDos). "
        "Uses `artifacts/model_bundle.joblib` if present, else `xg_boost_best_model.pkl` + meta."
    )

    kind, backend = get_backend()

    with st.sidebar:
        st.header("Model")
        if backend is None:
            st.warning("No model found. Train a bundle or add the pickle + meta.json.")
        elif kind == "bundle":
            st.success("Loaded: **XGBoost pipeline bundle**")
            if backend.meta:
                st.json(backend.meta)
        else:
            st.success("Loaded: **pickled sklearn model** (RandomForest)")
            st.caption("Features from `xg_boost_best_model.meta.json` (Friday CSV column order).")
        st.divider()
        st.markdown("**Default training data**")
        st.code(str(DEFAULT_CSV.name), language="text")
        st.markdown(
            "**Train bundle (XGBoost pipeline)**\n```bash\npython train_model.py --quick\n```"
        )
        st.markdown(
            "**AWS:** Docker image — see `AWS_DEPLOYMENT.md` (port **8080**)."
        )

    tab_a, tab_b = st.tabs(["Predict from CSV", "Train bundle (UI)"])

    with tab_a:
        if backend is None:
            st.info(
                "Add `artifacts/model_bundle.joblib` (run `python train_model.py --quick`) "
                "or `xg_boost_best_model.pkl` + `xg_boost_best_model.meta.json`."
            )
        up = st.file_uploader("Upload flow CSV (may omit Label for inference)", type=["csv"])
        if backend is not None and up is not None:
            df = pd.read_csv(up)
            df.columns = df.columns.str.strip()
            try:
                y_true = df[LABEL_COL] if LABEL_COL in df.columns else None

                if kind == "bundle":
                    assert isinstance(backend, ModelBundle)
                    X_proc = preprocess_for_predict(df, backend)
                    pred = backend.classifier.predict(X_proc)
                    proba = backend.classifier.predict_proba(X_proc)
                    out_labels = backend.label_encoder.inverse_transform(pred)
                else:
                    assert isinstance(backend, SklearnPklBackend)
                    pred, proba, out_labels = backend.predict_all(df)

                out = df.copy()
                out["predicted_label"] = out_labels
                if proba.shape[1] >= 2:
                    out["prediction_confidence"] = np.max(proba, axis=1)
                st.dataframe(out.head(50), use_container_width=True)
                if y_true is not None:
                    from sklearn.metrics import accuracy_score, f1_score

                    y_s = y_true.astype(str).str.strip()
                    p_s = pd.Series(out_labels).astype(str).str.strip()
                    if kind == "bundle":
                        mask = y_s.isin(backend.label_encoder.classes_)
                        if mask.all():
                            y_enc = backend.label_encoder.transform(y_s)
                            c1, c2 = st.columns(2)
                            c1.metric("Accuracy", f"{accuracy_score(y_enc, pred):.4f}")
                            c2.metric("F1 (macro)", f"{f1_score(y_enc, pred, average='macro'):.4f}")
                        else:
                            st.warning("Some labels are not in the training encoder; skipped metrics.")
                    else:
                        c1, c2 = st.columns(2)
                        c1.metric("Accuracy", f"{accuracy_score(y_s, p_s):.4f}")
                        c2.metric("F1 (macro)", f"{f1_score(y_s, p_s, average='macro'):.4f}")
                st.download_button(
                    "Download predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="ddos_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")

    with tab_b:
        st.write(
            "Train the **notebook-style XGBoost bundle** on a CSV you upload. "
            f"CLI default file: `{DEFAULT_CSV}`."
        )
        train_file = st.file_uploader("Labeled training CSV", type=["csv"], key="train")
        quick = st.checkbox("Quick train (recommended in Streamlit)", value=True)
        grid = st.checkbox("GridSearchCV (slow)", value=False)
        if st.button("Train & save bundle") and train_file is not None:
            df = pd.read_csv(train_file)
            with st.spinner("Training…"):
                b = fit_bundle_from_raw(df, use_grid_search=grid, quick_train=quick)
                ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(b, ARTIFACT)
                get_backend.clear()
            st.success(f"Saved to `{ARTIFACT}`. Bundle is preferred on reload (`MODEL_BACKEND=auto`).")
            st.json(b.meta)


if __name__ == "__main__":
    main()
