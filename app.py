"""
DDoS Detection — main Streamlit app (CICIDS 2017-style flow features).
Enhanced: dark UI, prediction count cards, SHAP interpretability.
Docker / AWS use the same module (see Dockerfile).
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ddos_pipeline import LABEL_COL, ModelBundle, fit_bundle_from_raw, preprocess_for_predict
from model_loader import load_prediction_backend
from pkl_backend import SklearnPklBackend

ARTIFACT = Path(os.environ.get("MODEL_PATH", "artifacts/model_bundle.joblib"))
DEFAULT_CSV = (
    Path(__file__).resolve().parent / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #252d40; }
.count-card {
    border-radius: 12px; padding: 20px 24px;
    display: flex; flex-direction: column; gap: 4px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 8px;
}
.count-card.benign { background: linear-gradient(135deg,#0d3b2e,#0a4d3c); border: 1px solid #1a6b4f; }
.count-card.attack { background: linear-gradient(135deg,#3b0d0d,#4d0a0a); border: 1px solid #7a1a1a; }
.count-card.total  { background: linear-gradient(135deg,#1a1f35,#1e2540); border: 1px solid #3a4080; }
.count-label { font-size:12px; font-weight:600; letter-spacing:1px; text-transform:uppercase; opacity:.75; }
.count-value { font-size:36px; font-weight:700; line-height:1.1; }
.count-card.benign .count-value { color:#2ecc71; }
.count-card.attack .count-value { color:#e74c3c; }
.count-card.total  .count-value { color:#7289da; }
.count-sub { font-size:13px; opacity:.6; margin-top:2px; }
.section-header {
    font-size:13px; font-weight:700; letter-spacing:1.2px;
    text-transform:uppercase; color:#7289da;
    margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #252d40;
}
.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-ok  { background:#0d3b2e; color:#2ecc71; border:1px solid #2ecc71; }
.badge-err { background:#3b0d0d; color:#e74c3c; border:1px solid #e74c3c; }
.shap-box { background:#161b27; border:1px solid #252d40; border-radius:12px; padding:20px; margin-top:16px; }
.shap-title    { font-size:15px; font-weight:700; color:#ffffff; margin-bottom:4px; }
.shap-subtitle { font-size:12px; color:#8892b0; margin-bottom:16px; }
</style>
"""

_BENIGN_KW = {"BENIGN", "NORMAL", "0"}


@st.cache_resource
def get_backend():
    return load_prediction_backend()


def _split_counts(out_labels):
    labels = pd.Series(out_labels).astype(str).str.strip().str.upper()
    benign = int(labels.isin(_BENIGN_KW).sum())
    attack = len(labels) - benign
    return benign, attack, len(labels)


def render_count_cards(out_labels) -> None:
    benign, attack, total = _split_counts(out_labels)
    attack_pct = 100 * attack / total if total else 0
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="count-card benign">'
            '<div class="count-label">&#10003; Benign Traffic</div>'
            f'<div class="count-value">{benign:,}</div>'
            f'<div class="count-sub">{100 - attack_pct:.1f}% of all flows</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="count-card attack">'
            '<div class="count-label">&#9888; DDoS / Attack</div>'
            f'<div class="count-value">{attack:,}</div>'
            f'<div class="count-sub">{attack_pct:.1f}% of all flows</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="count-card total">'
            '<div class="count-label">&#9642; Total Flows</div>'
            f'<div class="count-value">{total:,}</div>'
            '<div class="count-sub">Processed in this batch</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)


def render_donut(out_labels):
    benign, attack, _ = _split_counts(out_labels)
    fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.pie(
        [benign, attack],
        colors=["#2ecc71", "#e74c3c"],
        explode=(0.03, 0.03),
        startangle=90,
        wedgeprops=dict(width=0.48, edgecolor="#0f1117", linewidth=2),
        autopct="%1.1f%%",
        pctdistance=0.75,
        textprops=dict(color="white", fontsize=10, fontweight="bold"),
    )
    patches = [
        mpatches.Patch(color="#2ecc71", label=f"Benign ({benign:,})"),
        mpatches.Patch(color="#e74c3c", label=f"Attack ({attack:,})"),
    ]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False, fontsize=9, labelcolor="white")
    ax.set_title("Prediction Distribution", color="white", fontsize=12, fontweight="bold", pad=8)
    plt.tight_layout()
    return fig


def _label_style(val):
    if str(val).strip().upper() in _BENIGN_KW:
        return "background-color:#0d3b2e;color:#2ecc71;font-weight:600"
    return "background-color:#3b0d0d;color:#e74c3c;font-weight:600"


def render_shap(backend, kind: str, X_proc, n_samples: int = 5) -> None:
    st.markdown('<div class="shap-box">', unsafe_allow_html=True)
    st.markdown('<div class="shap-title">Model Interpretability - SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="shap-subtitle">'
        "How the model interprets each feature to arrive at a prediction (XGBoost SHAP values)"
        "</div>",
        unsafe_allow_html=True,
    )
    try:
        import shap  # noqa: PLC0415

        clf = (
            backend.classifier
            if hasattr(backend, "classifier")
            else getattr(backend, "model", backend)
        )
        sample = X_proc.iloc[:n_samples] if hasattr(X_proc, "iloc") else X_proc[:n_samples]
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(sample)

        sv = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        feat_names = (
            list(X_proc.columns)
            if hasattr(X_proc, "columns")
            else [f"f{i}" for i in range(sv.shape[1])]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Global Feature Importance** (mean |SHAP|)")
            mean_abs = np.abs(sv).mean(axis=0)
            top_n = min(15, len(feat_names))
            idx = np.argsort(mean_abs)[-top_n:]
            med = np.median(mean_abs[idx])
            colors = ["#7289da" if mean_abs[i] > med else "#4a5a8a" for i in idx]
            fig_bar, ax = plt.subplots(figsize=(5, 4), facecolor="#161b27")
            ax.set_facecolor("#161b27")
            ax.barh([feat_names[i][:25] for i in idx], mean_abs[idx],
                    color=colors, edgecolor="none", height=0.65)
            ax.set_xlabel("mean |SHAP value|", color="#8892b0", fontsize=9)
            ax.tick_params(colors="#c0c8d8", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#252d40")
            fig_bar.patch.set_facecolor("#161b27")
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        with col2:
            st.markdown("**Per-Sample Feature Contributions** (row 0)")
            sv0 = sv[0]
            top_pos = np.argsort(sv0)[-8:]
            top_neg = np.argsort(sv0)[:8]
            all_idx = np.concatenate([top_neg, top_pos])
            bar_colors = ["#e74c3c" if sv0[i] < 0 else "#2ecc71" for i in all_idx]
            fig_wf, ax2 = plt.subplots(figsize=(5, 4), facecolor="#161b27")
            ax2.set_facecolor("#161b27")
            ax2.barh([feat_names[i][:25] for i in all_idx], sv0[all_idx],
                     color=bar_colors, edgecolor="none", height=0.65)
            ax2.axvline(0, color="#8892b0", linewidth=0.8)
            ax2.set_xlabel("SHAP value (sample 0)", color="#8892b0", fontsize=9)
            ax2.tick_params(colors="#c0c8d8", labelsize=8)
            for spine in ax2.spines.values():
                spine.set_edgecolor("#252d40")
            fig_wf.patch.set_facecolor("#161b27")
            plt.tight_layout()
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

        top_feat = feat_names[int(np.argmax(mean_abs))]
        st.info(
            "**Most influential feature:** `" + top_feat + "`  |  "
            "Green bars push toward **Attack**; red bars push toward **Benign**. "
            "The global chart ranks features by average impact across all sampled rows."
        )

    except ImportError:
        st.warning(
            "shap is not installed. Run: pip install shap  "
            "SHAP (SHapley Additive exPlanations) explains how each feature "
            "influences the prediction for individual network flows."
        )
    except Exception as exc:
        st.warning(f"SHAP explanation could not be generated: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="DDoS Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        "<div style='display:flex;align-items:center;gap:14px;margin-bottom:4px;'>"
        "<span style='font-size:36px;'>🛡️</span>"
        "<div>"
        "<div style='font-size:26px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;'>"
        "Cloud Shield AI: A Distributed, AI-Powered Framework for Proactive DDoS Detection and Cloud-Scale Threat Intelligence</div>"
        "<div style='font-size:13px;color:#8892b0;margin-top:2px;'>"
        "CICIDS-2017 &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; SHAP Interpretability</div>"
        "</div></div>"
        "<hr style='border:none;border-top:1px solid #252d40;margin:12px 0 24px 0;'>",
        unsafe_allow_html=True,
    )

    kind, backend = get_backend()

    with st.sidebar:
        st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)
        if backend is None:
            st.error("No model loaded")
            st.info("Train a bundle or add the pickle + meta.json")
        elif kind == "bundle":
            st.markdown('<span class="badge badge-ok">BUNDLE LOADED</span>', unsafe_allow_html=True)
            st.markdown("**XGBoost Pipeline Bundle**")
            if backend.meta:
                with st.expander("Model metadata"):
                    st.json(backend.meta)
        else:
            st.markdown('<span class="badge badge-ok">PKL LOADED</span>', unsafe_allow_html=True)
            st.markdown("**Pickled sklearn / XGBoost**")
            st.caption("Features from matching `*.meta.json`")

        st.divider()
        st.markdown('<div class="section-header">Data</div>', unsafe_allow_html=True)
        st.code(DEFAULT_CSV.name, language="text")

        st.divider()
        st.markdown('<div class="section-header">Quick Commands</div>', unsafe_allow_html=True)
        st.code("streamlit run streamlit_app.py", language="bash")
        st.code("python train_model.py --quick", language="bash")

        st.divider()
        st.caption("AWS deployment via Docker on port **8080** — see AWS_DEPLOYMENT.md")

    tab_a, tab_b = st.tabs(["🔍  Predict from CSV", "🏋️  Train Bundle (UI)"])

    # ---------- Tab A: Predict ------------------------------------------------
    with tab_a:
        if backend is None:
            st.info(
                "Add artifacts/model_bundle.joblib (run: python train_model.py --quick) "
                "or xg_boost_best_model.pkl + xg_boost_best_model.meta.json"
            )
        else:
            st.markdown(
                "Upload a CICIDS-style flow CSV. The **Label** column is optional — "
                "if present, accuracy metrics will be computed automatically."
            )
            up = st.file_uploader("Upload flow CSV", type=["csv"], label_visibility="collapsed")

            if up is not None:
                df = pd.read_csv(up)
                df.columns = df.columns.str.strip()
                try:
                    y_true = df[LABEL_COL] if LABEL_COL in df.columns else None

                    with st.spinner("Running inference…"):
                        if kind == "bundle" and hasattr(backend, "classifier"):
                            X_proc = preprocess_for_predict(df, backend)
                            pred = backend.classifier.predict(X_proc)
                            proba = backend.classifier.predict_proba(X_proc)
                            out_labels = backend.label_encoder.inverse_transform(pred)
                        elif hasattr(backend, "predict_all"):
                            pred, proba, out_labels = backend.predict_all(df)
                            X_proc = df
                        else:
                            raise RuntimeError(
                                f"Backend type '{type(backend).__name__}' is not supported. "
                                "Expected a ModelBundle or SklearnPklBackend."
                            )

                    out = df.copy()
                    out["predicted_label"] = out_labels
                    if proba.shape[1] >= 2:
                        out["prediction_confidence"] = np.max(proba, axis=1)

                    st.markdown('<div class="section-header">Prediction Summary</div>', unsafe_allow_html=True)
                    render_count_cards(out_labels)

                    col_chart, col_metrics = st.columns([1, 1])
                    with col_chart:
                        fig_donut = render_donut(out_labels)
                        st.pyplot(fig_donut, use_container_width=True)
                        plt.close(fig_donut)

                    with col_metrics:
                        if y_true is not None:
                            from sklearn.metrics import (  # noqa: PLC0415
                                accuracy_score, classification_report, f1_score,
                            )
                            y_s = y_true.astype(str).str.strip()
                            p_s = pd.Series(out_labels).astype(str).str.strip()
                            st.markdown('<div class="section-header">Ground Truth Metrics</div>', unsafe_allow_html=True)
                            if kind == "bundle":
                                mask = y_s.isin(backend.label_encoder.classes_)
                                if mask.all():
                                    y_enc = backend.label_encoder.transform(y_s)
                                    m1, m2 = st.columns(2)
                                    m1.metric("Accuracy", f"{accuracy_score(y_enc, pred):.4f}")
                                    m2.metric("F1 (macro)", f"{f1_score(y_enc, pred, average='macro'):.4f}")
                                    with st.expander("Classification report"):
                                        st.text(classification_report(
                                            y_enc, pred,
                                            target_names=backend.label_encoder.classes_,
                                        ))
                                else:
                                    st.warning("Some labels not in encoder — metrics skipped.")
                            else:
                                m1, m2 = st.columns(2)
                                m1.metric("Accuracy", f"{accuracy_score(y_s, p_s):.4f}")
                                m2.metric("F1 (macro)", f"{f1_score(y_s, p_s, average='macro'):.4f}")
                        else:
                            st.info("Upload a CSV with a **Label** column to see accuracy metrics.")

                    st.divider()
                    st.markdown('<div class="section-header">Prediction Results (first 50 rows)</div>', unsafe_allow_html=True)
                    styled = out.head(50).style.map(_label_style, subset=["predicted_label"])
                    st.dataframe(styled, use_container_width=True, height=340)

                    st.download_button(
                        label="Download predictions CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="ddos_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.divider()
                    with st.expander("🔍  Model Interpretability — SHAP Analysis", expanded=False):
                        n_shap = st.slider("Samples for SHAP", 1, min(50, len(X_proc)), 5)
                        if st.button("Generate SHAP Explanation"):
                            with st.spinner("Computing SHAP values…"):
                                render_shap(backend, kind, X_proc, n_samples=n_shap)

                except Exception as exc:  # noqa: BLE001
                    st.error(f"Prediction failed: {exc}")
                    st.exception(exc)

    # ---------- Tab B: Train --------------------------------------------------
    with tab_b:
        st.write(
            "Train the **XGBoost pipeline bundle** on a labeled CSV you upload here. "
            f"CLI default file: {DEFAULT_CSV.name}"
        )
        train_file = st.file_uploader("Labeled training CSV", type=["csv"], key="train")

        col1, col2 = st.columns(2)
        with col1:
            quick = st.checkbox("Quick train (recommended in Streamlit)", value=True)
        with col2:
            grid = st.checkbox("GridSearchCV (slow)", value=False)

        if grid:
            st.warning("GridSearchCV may take 10-30 minutes depending on dataset size.")

        if st.button("Train & Save Bundle", use_container_width=True, type="primary"):
            if train_file is None:
                st.warning("Please upload a CSV file first.")
            else:
                df_train = pd.read_csv(train_file)
                progress = st.progress(0, text="Loading data…")
                with st.spinner("Training in progress…"):
                    progress.progress(20, text="Pre-processing features…")
                    b = fit_bundle_from_raw(df_train, use_grid_search=grid, quick_train=quick)
                    progress.progress(80, text="Saving bundle…")
                    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(b, ARTIFACT)
                    get_backend.clear()
                    progress.progress(100, text="Done!")
                st.success(f"Bundle saved to {ARTIFACT}. Reload the app to use it.")
                st.balloons()
                with st.expander("Bundle metadata"):
                    st.json(b.meta)


if __name__ == "__main__":
    main()
