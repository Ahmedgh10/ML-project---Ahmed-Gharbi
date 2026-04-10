from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load
from sklearn.neighbors import NearestNeighbors

from .data import make_features, explain_feature_columns


@dataclass(frozen=True)
class LoadedAppArtifacts:
    artifacts: Any
    normal_vectors: np.ndarray
    normal_explain: pd.DataFrame
    nn: NearestNeighbors


def load_app_artifacts(artifact_dir: str | Path) -> LoadedAppArtifacts:
    artifact_dir = Path(artifact_dir)
    artifacts = load(artifact_dir / "model.joblib")

    normal_vectors = np.load(artifact_dir / "normal_vectors.npy")
    normal_explain = pd.read_csv(artifact_dir / "normal_explain.csv")

    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(normal_vectors)

    return LoadedAppArtifacts(
        artifacts=artifacts,
        normal_vectors=normal_vectors,
        normal_explain=normal_explain,
        nn=nn,
    )


def predict_with_explanations(
    *,
    loaded: LoadedAppArtifacts,
    raw_df: pd.DataFrame,
    use_device_dynamics: bool,
    top_delta_features: int = 8,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Return predictions dataframe and per-row explanation tables.

    Explanations are only computed for predicted anomalies.
    """

    feats = make_features(raw_df, use_device_dynamics=use_device_dynamics)

    # The pipeline was trained on features without label
    X = feats.copy()
    if "label" in X.columns:
        X = X.drop(columns=["label"])

    pipe = loaded.artifacts.pipeline

    preds = pipe.predict(X)
    proba: np.ndarray | None = None
    class_labels: list[str] | None = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        try:
            class_labels = [str(c) for c in pipe.named_steps["clf"].classes_]
        except Exception:
            class_labels = None

    out = pd.DataFrame({
        "row_id": np.arange(len(raw_df), dtype=int),
        "timestamp": raw_df["timestamp"].astype(str),
        "device_id": raw_df["device_id"].astype(str),
        "device_type": raw_df["device_type"].astype(str),
        "predicted_label": preds,
    })

    if proba is not None:
        confidence = proba.max(axis=1)
        out["confidence"] = confidence

        if class_labels is not None and len(class_labels) == proba.shape[1]:
            for j, lbl in enumerate(class_labels):
                out[f"proba_{lbl}"] = proba[:, j]

    explanations: dict[int, pd.DataFrame] = {}

    preprocess = pipe.named_steps["preprocess"]
    Z = preprocess.transform(X)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()

    explain_cols = explain_feature_columns(use_device_dynamics=use_device_dynamics)

    for i, pred in enumerate(preds):
        if str(pred) == "Normal":
            continue

        distances, indices = loaded.nn.kneighbors(Z[i : i + 1], n_neighbors=1)
        nn_idx = int(indices[0, 0])
        twin = loaded.normal_explain.iloc[nn_idx]

        # Build delta table using explain columns available.
        deltas = []
        for col in explain_cols:
            if col in feats.columns and col in twin.index:
                try:
                    a = float(feats.iloc[i][col])
                    b = float(twin[col])
                except Exception:
                    continue
                deltas.append((col, a, b, a - b))

        delta_df = pd.DataFrame(deltas, columns=["feature", "value", "normal_twin", "delta"])
        delta_df["abs_delta"] = delta_df["delta"].abs()
        delta_df = delta_df.sort_values("abs_delta", ascending=False).head(top_delta_features)
        delta_df = delta_df.drop(columns=["abs_delta"]).reset_index(drop=True)

        # Attach metadata
        meta = pd.DataFrame(
            {
                "twin_device_id": [str(twin.get("device_id", ""))],
                "twin_device_type": [str(twin.get("device_type", ""))],
                "twin_timestamp": [str(twin.get("timestamp", ""))],
                "distance": [float(distances[0, 0])],
            }
        )

        explanations[i] = (meta, delta_df)

    return out, explanations
