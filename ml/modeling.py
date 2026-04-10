from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass(frozen=True)
class TrainConfig:
    use_device_dynamics: bool = False
    include_device_id_as_feature: bool = False
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class Artifacts:
    pipeline: Pipeline
    labels: list[str]
    config: TrainConfig


def build_preprocessor(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models(random_state: int) -> dict[str, Any]:
    return {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
    }


def evaluate_model(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
    pred = pipeline.predict(X_val)
    macro_f1 = f1_score(y_val, pred, average="macro")
    report = classification_report(y_val, pred, digits=4)
    return {"macro_f1": float(macro_f1), "report": report}


def train_and_select(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    device_ids: pd.Series,
    config: TrainConfig,
    numeric_features: list[str],
    categorical_features: list[str],
    regime: str,
) -> tuple[Artifacts, dict[str, Any]]:
    if regime not in {"random", "time", "group"}:
        raise ValueError("regime must be one of: random, time, group")

    if regime == "time":
        # Forward-in-time split
        order = np.argsort(pd.to_datetime(X["timestamp"]).values)
        split = int((1.0 - config.test_size) * len(order))
        train_idx = order[:split]
        val_idx = order[split:]
    elif regime == "group":
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        train_idx, val_idx = next(splitter.split(X, y, groups=device_ids))
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        train_idx, val_idx = next(splitter.split(X, y))

    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    models = candidate_models(config.random_state)

    best_name: str | None = None
    best_score = -1.0
    best_pipe: Pipeline | None = None
    metrics_by_model: dict[str, Any] = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_val, y_val)
        metrics_by_model[name] = metrics
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_name = name
            best_pipe = pipe

    assert best_name is not None and best_pipe is not None

    # Refit best on combined train+val for a single artifact
    best_pipe.fit(X, y)

    labels = sorted(y.unique().tolist())
    artifacts = Artifacts(pipeline=best_pipe, labels=labels, config=config)
    summary = {"best_model": best_name, "best_macro_f1": best_score, "models": metrics_by_model}
    return artifacts, summary


def build_normal_twin_index(
    *,
    pipeline: Pipeline,
    X_features: pd.DataFrame,
    raw_meta: pd.DataFrame,
    y: pd.Series,
    explain_cols: list[str],
    max_normals: int = 20000,
) -> dict[str, Any]:
    normal_mask = y.astype(str).eq("Normal")
    X_norm = X_features.loc[normal_mask]
    meta_norm = raw_meta.loc[normal_mask]

    if len(X_norm) == 0:
        raise ValueError("No Normal samples available to build normal-twin index")

    if len(X_norm) > max_normals:
        X_norm = X_norm.sample(n=max_normals, random_state=0)
        meta_norm = meta_norm.loc[X_norm.index]

    preprocess = pipeline.named_steps["preprocess"]
    Z = preprocess.transform(X_norm)

    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(Z)

    explain_frame = meta_norm.copy()
    for col in explain_cols:
        if col in X_features.columns:
            explain_frame[col] = X_features.loc[explain_frame.index, col]

    return {
        "normal_vectors": Z,
        "normal_explain": explain_frame,
        "nn": nn,
    }


def save_artifacts(
    *,
    artifact_dir: str,
    artifacts: Artifacts,
    normal_vectors: Any,
    normal_explain: pd.DataFrame,
) -> None:
    from pathlib import Path

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dump(artifacts, out_dir / "model.joblib")

    # Store vectors as .npy (dense) if possible; otherwise as sparse npz-ish array
    # sklearn transformers may output sparse; NearestNeighbors expects dense.
    if hasattr(normal_vectors, "toarray"):
        normal_vectors = normal_vectors.toarray()

    np.save(out_dir / "normal_vectors.npy", np.asarray(normal_vectors))
    normal_explain.to_csv(out_dir / "normal_explain.csv", index=False)
