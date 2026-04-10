from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.data import load_csv, validate_input_frame, split_X_y, make_features, explain_feature_columns
from ml.modeling import TrainConfig, train_and_select, build_normal_twin_index, save_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--artifact-dir", default="artifacts", help="Where to write artifacts")
    parser.add_argument(
        "--regime",
        default="random",
        choices=["random", "time", "group"],
        help="Validation split regime",
    )
    parser.add_argument(
        "--use-device-dynamics",
        action="store_true",
        help="Add lag/delta features per device_id",
    )
    parser.add_argument(
        "--include-device-id",
        action="store_true",
        help="Include device_id as a categorical feature (can hurt unseen-device generalization)",
    )
    args = parser.parse_args()

    df = load_csv(args.data)
    validate_input_frame(df, require_label=True)

    dataset = split_X_y(df, require_label=True)

    features = make_features(
        pd.concat([dataset.X, dataset.y], axis=1),
        use_device_dynamics=bool(args.use_device_dynamics),
    )

    y = features["label"].astype(str)
    X = features.drop(columns=["label"])

    # Build feature lists
    categorical_features = ["device_type"]
    if args.include_device_id:
        categorical_features.append("device_id")

    numeric_features = [
        c
        for c in X.columns
        if c
        not in {
            "timestamp",
            "device_id",
            "device_type",
        }
    ]

    config = TrainConfig(
        use_device_dynamics=bool(args.use_device_dynamics),
        include_device_id_as_feature=bool(args.include_device_id),
    )

    artifacts, summary = train_and_select(
        X=X,
        y=y,
        device_ids=X["device_id"].astype(str),
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        regime=args.regime,
    )

    explain_cols = explain_feature_columns(use_device_dynamics=config.use_device_dynamics)

    raw_meta = X[["timestamp", "device_id", "device_type"]].copy()
    normal = build_normal_twin_index(
        pipeline=artifacts.pipeline,
        X_features=X,
        raw_meta=raw_meta,
        y=y,
        explain_cols=explain_cols,
    )

    save_artifacts(
        artifact_dir=args.artifact_dir,
        artifacts=artifacts,
        normal_vectors=normal["normal_vectors"],
        normal_explain=normal["normal_explain"],
    )

    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved artifacts to:", str(out_dir.resolve()))
    print("Best model:", summary["best_model"], "macro_f1:", summary["best_macro_f1"])


if __name__ == "__main__":
    main()
