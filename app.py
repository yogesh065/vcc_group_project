
"""
Streamlit UI for DDoS traffic detection (CICIDS 2017-style flow features).
Enhanced with: better UI, prediction count cards, SHAP explanation for XGBoost.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ddos_pipeline import LABEL_COL, ModelBundle, fit_bundle_from_raw, preprocess_for_predict
from model_loader import load_prediction_backend
from pkl_backend import SklearnPklBackend

ARTIFACT = Path(os.environ.get("MODEL_PATH", "artifacts/model_bundle.joblib"))
DEFAULT_CSV = Path(__file__).resolve().parent / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #0f1117;
}

/* ── Metric / count cards ── */
.count-card {
    border-radius: 12px;
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.count-card.benign {
    background: linear-gradient(135deg, #0d3b2e 0%, #0a4d3c 100%);
    border: 1px solid #1a6b4f;
}
.count-card.attack {
    background: linear-gradient(135deg, #3b0d0d 0%, #4d0a0a 100%);
    border: 1px solid #7a1a1a;
}
.count-card.total {
    background: linear-gradient(135deg, #1a1f35 0%, #1e2540 100%);
    border: 1px solid #3a4080;
}
.count-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity: 0.75;
}
.count-value {
    font-size: 36px;
    font-weight: 700;
    line-height: 1.1;
}
.count-card.benign .count-value { color: #2ecc71; }
.count-card.attack .count-value { color: #e74c3c; }
.count-card.total .count-value { color: #7289da; }
.count-sub {
    font-size: 13px;
    opacity: 0.6;
    margin-top: 2px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #252d40;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.3px;
}

/* ── Section headers ── */
.section-header {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #7289da;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #252d40;
}

/* ── Status badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-ok  { background: #0d3b2e; color: #2ecc71; border: 1px solid #2ecc71; }
.badge-err { background: #3b0d0d; color: #e74c3c; border: 1px solid #e74c3c; }

/* ── SHAP explanation box ── */
.shap-box {
    background: #161b27;
    border: 1px solid #252d40;
    border-radius: 12px;
    padding: 20px;
    margin-top: 16px;
}
.shap-title {
    font-size: 15px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}
.shap-subtitle {
    font-size: 12px;
    color: #8892b0;
    margin-bottom: 16px;
}
</style>
"""

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

@st.cache_resource
def get_backend():
    return load_prediction_backend()


def render_count_cards(out_labels: np.ndarray | list) -> None:
    """Render Benign / Attack / Total prediction count cards."""
    labels = pd.Series(out_labels).astype(str).str.strip().str.upper()
    benign_kw = ["BENIGN", "NORMAL", "0"]
    benign_count = labels.isin(benign_kw).sum()
    attack_count = len(labels) - benign_count
    total_count  = len(labels)
    attack_pct   = 100 * attack_count / total_count if total_count else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="count-card benign">
            <div class="count-label">✅ Benign Traffic</div>
            <div class="count-value">{benign_count:,}</div>
            <div class="count-sub">{100 - attack_pct:.1f}% of all flows</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="count-card attack">
            <div class="count-label">🚨 DDoS / Attack</div>
            <div class="count-value">{attack_count:,}</div>
            <div class="count-sub">{attack_pct:.1f}% of all flows</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="count-card total">
            <div class="count-label">📊 Total Flows</div>
            <div class="count-value">{total_count:,}</div>
            <div class="count-sub">Processed in this batch</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def render_prediction_donut(out_labels: np.ndarray | list) -> None:
    """Donut chart showing Benign vs Attack ratio."""
    labels = pd.Series(out_labels).astype(str).str.strip().str.upper()
    benign_kw = ["BENIGN", "NORMAL", "0"]
    benign_count = labels.isin(benign_kw).sum()
    attack_count = len(labels) - benign_count

    fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor='#0f1117')
    ax.set_facecolor('#0f1117')
    sizes  = [benign_count, attack_count]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.03, 0.03)
    wedge_props = dict(width=0.48, edgecolor='#0f1117', linewidth=2)
    ax.pie(sizes, colors=colors, explode=explode, startangle=90,
           wedgeprops=wedge_props, autopct='%1.1f%%',
           pctdistance=0.75,
           textprops=dict(color='white', fontsize=10, fontweight='bold'))
    benign_patch = mpatches.Patch(color='#2ecc71', label=f'Benign ({benign_count:,})')
    attack_patch = mpatches.Patch(color='#e74c3c', label=f'Attack ({attack_count:,})')
    ax.legend(handles=[benign_patch, attack_patch], loc='lower center',
              bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False, fontsize=9,
              labelcolor='white')
    ax.set_title('Prediction Distribution', color='white', fontsize=12, fontweight='bold', pad=8)
    plt.tight_layout()
    return fig


def render_shap_explanation(backend, kind: str, X_proc: pd.DataFrame, n_samples: int = 5) -> None:
    """
    Render SHAP waterfall / bar plot for XGBoost predictions.
    Gracefully degrades if shap is not installed.
    """
    st.markdown('<div class="shap-box">', unsafe_allow_html=True)
    st.markdown('<div class="shap-title">🔍 Model Interpretability — SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="shap-subtitle">How the model interprets each feature to make a prediction (XGBoost SHAP values)</div>', unsafe_allow_html=True)

    try:
        import shap  # noqa: PLC0415

        clf = backend.classifier if kind == "bundle" else backend.model
        sample = X_proc.iloc[:n_samples] if hasattr(X_proc, 'iloc') else X_proc[:n_samples]

        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer(sample)

        # ── Bar chart: mean |SHAP| across samples (global feature importance)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Global Feature Importance** (mean |SHAP|)")
            fig_bar, ax = plt.subplots(figsize=(5, 4), facecolor='#161b27')
            ax.set_facecolor('#161b27')

            if hasattr(shap_values, 'values'):
                sv = shap_values.values
                # For multi-class take class 1 (attack), for binary take first output
                if sv.ndim == 3:
                    sv = sv[:, :, 1]
            else:
                sv = shap_values

            mean_abs = np.abs(sv).mean(axis=0)
            feat_names = list(X_proc.columns) if hasattr(X_proc, 'columns') else [f"f{i}" for i in range(mean_abs.shape[0])]
            top_n  = min(15, len(feat_names))
            idx    = np.argsort(mean_abs)[-top_n:]
            colors = ['#7289da' if v > np.median(mean_abs[idx]) else '#4a5a8a' for v in mean_abs[idx]]

            ax.barh([feat_names[i] for i in idx], mean_abs[idx], color=colors, edgecolor='none', height=0.65)
            ax.set_xlabel("mean |SHAP value|", color='#8892b0', fontsize=9)
            ax.tick_params(colors='#c0c8d8', labelsize=8)
            ax.spines[:].set_color('#252d40')
            ax.set_facecolor('#161b27')
            for spine in ax.spines.values():
                spine.set_edgecolor('#252d40')
            fig_bar.patch.set_facecolor('#161b27')
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        with col2:
            st.markdown("**Sample-level SHAP** (first row waterfall)")
            try:
                fig_wf = plt.figure(figsize=(5, 4), facecolor='#161b27')
                shap.plots.waterfall(shap_values[0] if hasattr(shap_values[0], 'values') else shap_values[0],
                                     max_display=10, show=False)
                fig_wf.patch.set_facecolor('#161b27')
                plt.tight_layout()
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)
            except Exception:
                # Fallback: show positive vs negative contributions
                sv0 = sv[0]
                pos_idx = np.argsort(sv0)[-8:]
                neg_idx = np.argsort(sv0)[:8]
                all_idx = np.concatenate([neg_idx, pos_idx])
                fig_wf2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#161b27')
                ax2.set_facecolor('#161b27')
                bar_colors = ['#e74c3c' if sv0[i] < 0 else '#2ecc71' for i in all_idx]
                ax2.barh([feat_names[i][:20] for i in all_idx], sv0[all_idx],
                         color=bar_colors, edgecolor='none', height=0.65)
                ax2.axvline(0, color='#8892b0', linewidth=0.8)
                ax2.set_xlabel("SHAP value (row 0)", color='#8892b0', fontsize=9)
                ax2.tick_params(colors='#c0c8d8', labelsize=8)
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#252d40')
                fig_wf2.patch.set_facecolor('#161b27')
                plt.tight_layout()
                st.pyplot(fig_wf2, use_container_width=True)
                plt.close(fig_wf2)

        # ── SHAP interpretation text
        if len(feat_names) > 0:
            top_feat = feat_names[np.argmax(mean_abs)]
            st.info(
                f"📌 **Most influential feature:** `{top_feat}` — "
                "Higher SHAP values (green) push the prediction toward **Attack**; "
                "lower (red) push toward **Benign**. "
                "The global bar chart shows which features matter most across all samples."
            )

    except ImportError:
        st.warning(
            "⚠️ `shap` is not installed. Run `pip install shap` to enable model interpretability.

"
            "SHAP (SHapley Additive exPlanations) shows how each feature contributes to "
            "the model's prediction for individual flows."
        )
    except Exception as exc:
        st.warning(f"SHAP explanation could not be generated: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="DDoS Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
        <span style="font-size:36px;">🛡️</span>
        <div>
            <div style="font-size:26px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">DDoS Traffic Detection</div>
            <div style="font-size:13px;color:#8892b0;margin-top:2px;">
                CICIDS-2017 · XGBoost · SHAP Interpretability
            </div>
        </div>
    </div>
    <hr style="border:none;border-top:1px solid #252d40;margin:12px 0 24px 0;">
    """, unsafe_allow_html=True)

    kind, backend = get_backend()

    # ── Sidebar ──────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Model Status</div>', unsafe_allow_html=True)

        if backend is None:
            st.error("No model loaded")
            st.info("Train a bundle or add the pickle + meta.json")
        elif kind == "bundle":
            st.markdown('<span class="badge badge-ok">BUNDLE LOADED</span>', unsafe_allow_html=True)
            st.markdown("**XGBoost Pipeline Bundle**", unsafe_allow_html=False)
            if backend.meta:
                with st.expander("Model metadata"):
                    st.json(backend.meta)
        else:
            st.markdown('<span class="badge badge-ok">PKL LOADED</span>', unsafe_allow_html=True)
            st.markdown("**Pickled sklearn / XGBoost**")
            st.caption("Features from matching `*.meta.json`")

        st.divider()
        st.markdown('<div class="section-header">📁 Data</div>', unsafe_allow_html=True)
        st.code(DEFAULT_CSV.name, language="text")

        st.divider()
        st.markdown('<div class="section-header">🚀 Quick Commands</div>', unsafe_allow_html=True)
        st.code("streamlit run streamlit_app.py", language="bash")
        st.code("python train_model.py --quick", language="bash")

        st.divider()
        st.caption("🐳 AWS deployment via Docker on port **8080** — see `AWS_DEPLOYMENT.md`")

    # ── Tabs ─────────────────────────────────────────
    tab_a, tab_b = st.tabs(["🔍 Predict from CSV", "🏋️ Train Bundle (UI)"])

    # ════════════════════════════════════════════════
    # Tab A — Predict
    # ════════════════════════════════════════════════
    with tab_a:
        if backend is None:
            st.info(
                "Add `artifacts/model_bundle.joblib` (run `python train_model.py --quick`) "
                "or `xg_boost_best_model.pkl` + `xg_boost_best_model.meta.json`."
            )
        else:
            st.markdown(
                "Upload a CICIDS-style flow CSV. The **Label** column is optional — "
                "if present, accuracy metrics will be computed automatically.",
                unsafe_allow_html=False,
            )
            up = st.file_uploader("📂 Upload flow CSV", type=["csv"], label_visibility="collapsed")

            if up is not None:
                df = pd.read_csv(up)
                df.columns = df.columns.str.strip()

                try:
                    y_true = df[LABEL_COL] if LABEL_COL in df.columns else None

                    with st.spinner("Running inference…"):
                        if kind == "bundle":
                            assert isinstance(backend, ModelBundle)
                            X_proc = preprocess_for_predict(df, backend)
                            pred   = backend.classifier.predict(X_proc)
                            proba  = backend.classifier.predict_proba(X_proc)
                            out_labels = backend.label_encoder.inverse_transform(pred)
                        else:
                            assert isinstance(backend, SklearnPklBackend)
                            pred, proba, out_labels = backend.predict_all(df)
                            X_proc = df  # used for SHAP

                    out = df.copy()
                    out["predicted_label"]      = out_labels
                    if proba.shape[1] >= 2:
                        out["prediction_confidence"] = np.max(proba, axis=1)

                    # ── Count cards ──────────────────────
                    st.markdown('<div class="section-header">📊 Prediction Summary</div>', unsafe_allow_html=True)
                    render_count_cards(out_labels)

                    # ── Donut + metrics ───────────────────
                    col_chart, col_metrics = st.columns([1, 1])
                    with col_chart:
                        fig_donut = render_prediction_donut(out_labels)
                        st.pyplot(fig_donut, use_container_width=True)
                        plt.close(fig_donut)

                    with col_metrics:
                        if y_true is not None:
                            from sklearn.metrics import accuracy_score, f1_score, classification_report  # noqa: PLC0415

                            y_s = y_true.astype(str).str.strip()
                            p_s = pd.Series(out_labels).astype(str).str.strip()

                            st.markdown('<div class="section-header">📈 Ground Truth Metrics</div>', unsafe_allow_html=True)

                            if kind == "bundle":
                                mask = y_s.isin(backend.label_encoder.classes_)
                                if mask.all():
                                    y_enc = backend.label_encoder.transform(y_s)
                                    acc   = accuracy_score(y_enc, pred)
                                    f1    = f1_score(y_enc, pred, average="macro")
                                    m1, m2 = st.columns(2)
                                    m1.metric("Accuracy",   f"{acc:.4f}")
                                    m2.metric("F1 (macro)", f"{f1:.4f}")
                                    with st.expander("Classification report"):
                                        st.text(classification_report(y_enc, pred,
                                                    target_names=backend.label_encoder.classes_))
                                else:
                                    st.warning("Some labels not in encoder — metrics skipped.")
                            else:
                                acc = accuracy_score(y_s, p_s)
                                f1  = f1_score(y_s, p_s, average="macro")
                                m1, m2 = st.columns(2)
                                m1.metric("Accuracy",   f"{acc:.4f}")
                                m2.metric("F1 (macro)", f"{f1:.4f}")
                        else:
                            st.info("Upload a CSV with a **Label** column to see accuracy metrics.")

                    # ── Results dataframe ─────────────────
                    st.divider()
                    st.markdown('<div class="section-header">🗂️ Prediction Results (first 50 rows)</div>', unsafe_allow_html=True)

                    # Colour-code predicted_label
                    def _style_label(val):
                        v = str(val).upper()
                        if v in ["BENIGN", "NORMAL", "0"]:
                            return "background-color:#0d3b2e;color:#2ecc71;font-weight:600"
                        return "background-color:#3b0d0d;color:#e74c3c;font-weight:600"

                    styled = (out.head(50)
                                 .style
                                 .applymap(_style_label, subset=["predicted_label"]))
                    st.dataframe(styled, use_container_width=True, height=340)

                    # ── Download ──────────────────────────
                    st.download_button(
                        label="⬇️ Download predictions CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="ddos_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    # ── SHAP Explanation ──────────────────
                    st.divider()
                    with st.expander("🔍 Model Interpretability — SHAP Analysis", expanded=True):
                        n_shap = st.slider("Number of samples for SHAP", 1, min(50, len(X_proc)), 5)
                        if st.button("Generate SHAP Explanation"):
                            with st.spinner("Computing SHAP values…"):
                                render_shap_explanation(backend, kind, X_proc, n_samples=n_shap)

                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ Prediction failed: {exc}")
                    st.exception(exc)

    # ════════════════════════════════════════════════
    # Tab B — Train
    # ════════════════════════════════════════════════
    with tab_b:
        st.markdown(
            "Train the **XGBoost pipeline bundle** on a labeled CSV you upload here. "
            f"CLI default file: `{DEFAULT_CSV.name}`."
        )
        train_file = st.file_uploader("📂 Labeled training CSV", type=["csv"], key="train")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            quick = st.checkbox("⚡ Quick train (recommended in Streamlit)", value=True)
        with col_opt2:
            grid = st.checkbox("🔬 GridSearchCV (slow)", value=False)

        if grid:
            st.warning("GridSearchCV may take 10-30 minutes depending on dataset size.")

        if st.button("🚀 Train & Save Bundle", use_container_width=True, type="primary") and train_file is not None:
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

            st.success(f"✅ Bundle saved to `{ARTIFACT}`. Reload the app to use it.")
            st.balloons()
            with st.expander("Bundle metadata"):
                st.json(b.meta)
        elif train_file is None and st.session_state.get("train_clicked"):
            st.warning("Please upload a CSV file first.")


if __name__ == "__main__":
    main()
