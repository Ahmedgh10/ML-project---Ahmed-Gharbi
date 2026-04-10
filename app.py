from __future__ import annotations

from io import StringIO
from pathlib import Path
import json

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, f1_score

from ml.data import load_csv, validate_input_frame
from ml.inference import load_app_artifacts, predict_with_explanations


st.set_page_config(page_title="Smart System Anomaly Detector", layout="wide")

st.title("Smart System Anomaly Detector")
st.caption("Upload telemetry rows → get predicted label + confidence, and a nearest-normal-twin explanation for anomalies.")

with st.sidebar:
    st.header("Artifacts")
    artifact_dir = st.text_input("Artifact directory", value="artifacts")

    summary_path = Path(artifact_dir) / "train_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            st.caption(
                f"Trained model: {summary.get('best_model', '?')} | best val macro_f1: {summary.get('best_macro_f1', '?')}"
            )
        except Exception:
            pass

    st.header("Input")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    use_device_dynamics = st.checkbox(
        "Use device dynamics (lag/delta features)",
        value=False,
        help="Only enable if the model was trained with --use-device-dynamics.",
    )

    show_probabilities = st.checkbox(
        "Show per-class probabilities",
        value=False,
        help="Adds proba_* columns if the model supports predict_proba.",
    )

    top_deltas = st.slider("Top deltas to show", min_value=3, max_value=15, value=8, step=1)

@st.cache_resource
def _load_cached(artifact_dir: str):
    return load_app_artifacts(artifact_dir)


if not Path(artifact_dir).exists():
    st.info("Train a model first (see README), then point to the artifacts directory.")
    st.stop()

try:
    loaded = _load_cached(artifact_dir)
except Exception as e:
    st.error(f"Failed to load artifacts from '{artifact_dir}': {e}")
    st.stop()

if uploaded is None:
    st.info("Upload a CSV file to see predictions.")
    st.stop()

try:
    content = uploaded.getvalue().decode("utf-8")
    raw_df = pd.read_csv(StringIO(content))

    # Parse timestamp consistently
    if "timestamp" in raw_df.columns:
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")

    validate_input_frame(raw_df, require_label=False)

    # If user uploaded with label, keep it for display but do not use for prediction
    has_label = "label" in raw_df.columns

except Exception as e:
    st.error(f"Invalid CSV: {e}")
    st.stop()

pred_df, explanations = predict_with_explanations(
    loaded=loaded,
    raw_df=raw_df,
    use_device_dynamics=use_device_dynamics,
    top_delta_features=int(top_deltas),
)

st.subheader("Predictions")

show = pred_df.copy()
if has_label:
    show.insert(4, "true_label", raw_df["label"].astype(str))

if not show_probabilities:
    proba_cols = [c for c in show.columns if c.startswith("proba_")]
    if proba_cols:
        show = show.drop(columns=proba_cols)

st.dataframe(show, use_container_width=True, hide_index=True)

st.subheader("Prediction summary")
counts = pred_df["predicted_label"].value_counts().rename_axis("label").reset_index(name="count")
st.bar_chart(counts.set_index("label")["count"], horizontal=True)

if has_label:
    st.subheader("Evaluation (uploaded labels)")
    y_true = raw_df["label"].astype(str)
    y_pred = pred_df["predicted_label"].astype(str)
    macro = f1_score(y_true, y_pred, average="macro")
    st.metric("Macro F1", value=f"{macro:.4f}")
    st.text(classification_report(y_true, y_pred, digits=4))

csv_bytes = show.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions as CSV",
    data=csv_bytes,
    file_name="predictions.csv",
    mime="text/csv",
)

anomaly_rows = [i for i in explanations.keys()]

st.subheader("Nearest-normal-twin explanation")
if not anomaly_rows:
    st.success("No anomalies predicted in this upload.")
    st.stop()

row_choice = st.selectbox(
    "Select an anomalous row to explain",
    options=anomaly_rows,
    format_func=lambda i: f"row_id={int(pred_df.iloc[i]['row_id'])} | pred={pred_df.iloc[i]['predicted_label']} | device={pred_df.iloc[i]['device_id']}",
)

meta_df, delta_df = explanations[int(row_choice)]

c1, c2 = st.columns([1, 2], gap="large")
with c1:
    st.markdown("**Twin metadata**")
    st.dataframe(meta_df, use_container_width=True, hide_index=True)
with c2:
    st.markdown("**Top feature deltas (value − normal_twin)**")
    st.dataframe(delta_df, use_container_width=True, hide_index=True)
