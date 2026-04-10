from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "device_id",
    "device_type",
    "cpu_usage",
    "memory_usage",
    "network_in_kb",
    "network_out_kb",
    "packet_rate",
    "avg_response_time_ms",
    "service_access_count",
    "failed_auth_attempts",
    "is_encrypted",
    "geo_location_variation",
    "label",
)


NUMERIC_COLS: tuple[str, ...] = (
    "cpu_usage",
    "memory_usage",
    "network_in_kb",
    "network_out_kb",
    "packet_rate",
    "avg_response_time_ms",
    "service_access_count",
    "failed_auth_attempts",
    "geo_location_variation",
)

CATEGORICAL_COLS: tuple[str, ...] = (
    "device_id",
    "device_type",
)

BINARY_COLS: tuple[str, ...] = ("is_encrypted",)


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if df["timestamp"].isna().any():
        bad = int(df["timestamp"].isna().sum())
        raise ValueError(f"Found {bad} invalid timestamp values")

    # Normalize label column name if present
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)

    return df


def validate_input_frame(df: pd.DataFrame, require_label: bool) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if not require_label:
        missing = [c for c in missing if c != "label"]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols_present = [c for c in NUMERIC_COLS if c in df.columns]
    if numeric_cols_present and df[numeric_cols_present].isna().any().any():
        # Keep it strict in training; in app we can impute.
        pass


def make_features(df: pd.DataFrame, *, use_device_dynamics: bool) -> pd.DataFrame:
    """Create ML features from raw columns.

    This function is deterministic and safe to run in both training and inference.

    Notes:
    - If `use_device_dynamics=True`, lag features are computed within the provided
      frame, grouped by device_id and ordered by timestamp.
    - For very short uploads (no previous row per device), lag features become NaN
      and should be handled by an imputer in the pipeline.
    """

    out = df.copy()

    out = out.sort_values("timestamp")

    out["hour"] = out["timestamp"].dt.hour.astype(int)
    out["dayofweek"] = out["timestamp"].dt.dayofweek.astype(int)

    eps = 1e-9
    out["total_traffic_kb"] = out["network_in_kb"] + out["network_out_kb"]
    out["in_out_ratio"] = out["network_in_kb"] / (out["network_out_kb"] + eps)
    out["packet_rate_per_kb"] = out["packet_rate"] / (out["total_traffic_kb"] + eps)

    if use_device_dynamics:
        group = out.groupby("device_id", sort=False)

        for col in (
            "cpu_usage",
            "memory_usage",
            "network_in_kb",
            "network_out_kb",
            "packet_rate",
            "avg_response_time_ms",
        ):
            out[f"lag1_{col}"] = group[col].shift(1)
            out[f"delta1_{col}"] = out[col] - out[f"lag1_{col}"]

    # Ensure types are consistent
    out["is_encrypted"] = pd.to_numeric(out["is_encrypted"], errors="coerce").astype("Int64")

    return out


def split_X_y(df: pd.DataFrame, *, require_label: bool) -> Dataset:
    if require_label and "label" not in df.columns:
        raise ValueError("label column is required")

    if require_label:
        y = df["label"].astype(str)
        X = df.drop(columns=["label"])
        return Dataset(X=X, y=y)

    # Dummy y for type compatibility
    return Dataset(X=df.copy(), y=pd.Series(np.zeros(len(df)), name="label"))


def explain_feature_columns(*, use_device_dynamics: bool) -> list[str]:
    base = [
        "cpu_usage",
        "memory_usage",
        "network_in_kb",
        "network_out_kb",
        "packet_rate",
        "avg_response_time_ms",
        "service_access_count",
        "failed_auth_attempts",
        "is_encrypted",
        "geo_location_variation",
        "hour",
        "dayofweek",
        "total_traffic_kb",
        "in_out_ratio",
        "packet_rate_per_kb",
    ]
    if use_device_dynamics:
        base.extend(
            [
                "delta1_cpu_usage",
                "delta1_memory_usage",
                "delta1_network_in_kb",
                "delta1_network_out_kb",
                "delta1_packet_rate",
                "delta1_avg_response_time_ms",
            ]
        )
    return base
