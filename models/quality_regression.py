from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_FEATURES = [
    "Quality 6",
    "Quality 7",
    "Quality 8",
    "Quality 9",
    "Quality 10",
    "Quality 11",
    "Quality 12",
    "Quality 13",
    "Quality 14",
    "Quality 15",
    "Quality 16",
]

MODEL_FEATURES = [
    *QUALITY_FEATURES,
    "previous_output",
    "previous_output_age_hours",
    "quality_age_hours",
    "hour_sin",
    "hour_cos",
]


def _wide_quality(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(
            index="Timestamp_Shifted",
            columns="Parameter",
            values="Value",
            aggfunc="mean",
        )
        .sort_index()
        .reset_index()
    )


def _attach_latest_features(
    targets: pd.DataFrame,
    feature_events: pd.DataFrame,
    output_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = pd.merge_asof(
        targets.sort_values("Timestamp_Shifted"),
        feature_events.sort_values("Timestamp_Shifted"),
        on="Timestamp_Shifted",
        direction="backward",
    )
    merged["quality_event_time"] = merged["quality_event_time"].ffill().bfill()
    merged[QUALITY_FEATURES] = merged[QUALITY_FEATURES].ffill().bfill()
    if output_events is not None and not output_events.empty:
        merged = pd.merge_asof(
            merged.sort_values("Timestamp_Shifted"),
            output_events.sort_values("Timestamp_Shifted"),
            on="Timestamp_Shifted",
            direction="backward",
            allow_exact_matches=False,
        )
        merged["previous_output"] = merged["previous_output"].ffill().bfill()
        merged["previous_output_time"] = merged["previous_output_time"].ffill().bfill()
        merged["previous_output_age_hours"] = (
            merged["Timestamp_Shifted"] - merged["previous_output_time"]
        ).dt.total_seconds().div(3600).clip(lower=0, upper=96)
    else:
        merged["previous_output"] = np.nan
        merged["previous_output_age_hours"] = np.nan

    merged["quality_age_hours"] = (
        merged["Timestamp_Shifted"] - merged["quality_event_time"]
    ).dt.total_seconds().div(3600).clip(lower=0, upper=96)
    hours = merged["Timestamp_Shifted"].dt.hour + (merged["Timestamp_Shifted"].dt.minute / 60)
    merged["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    merged["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    return merged


def build_training_frame(train_long: pd.DataFrame) -> pd.DataFrame:
    wide = _wide_quality(train_long)
    targets = wide.loc[wide["Output Parameter"].notna(), ["Timestamp_Shifted", "Output Parameter"]]
    feature_events = wide.loc[wide[QUALITY_FEATURES].notna().any(axis=1), ["Timestamp_Shifted", *QUALITY_FEATURES]].copy()
    feature_events["quality_event_time"] = feature_events["Timestamp_Shifted"]
    output_events = wide.loc[wide["Output Parameter"].notna(), ["Timestamp_Shifted", "Output Parameter"]].copy()
    output_events["previous_output"] = output_events["Output Parameter"]
    output_events["previous_output_time"] = output_events["Timestamp_Shifted"]
    output_events = output_events[["Timestamp_Shifted", "previous_output", "previous_output_time"]]
    frame = _attach_latest_features(targets, feature_events, output_events)
    return frame.dropna(subset=["Output Parameter", *MODEL_FEATURES]).reset_index(drop=True)


def build_prediction_frame(train_long: pd.DataFrame, test_long: pd.DataFrame, timestamps: pd.DataFrame) -> pd.DataFrame:
    available = pd.concat([train_long, test_long], ignore_index=True)
    wide = _wide_quality(available)
    feature_events = wide.loc[wide[QUALITY_FEATURES].notna().any(axis=1), ["Timestamp_Shifted", *QUALITY_FEATURES]].copy()
    feature_events["quality_event_time"] = feature_events["Timestamp_Shifted"]
    output_events = wide.loc[wide["Output Parameter"].notna(), ["Timestamp_Shifted", "Output Parameter"]].copy()
    output_events["previous_output"] = output_events["Output Parameter"]
    output_events["previous_output_time"] = output_events["Timestamp_Shifted"]
    output_events = output_events[["Timestamp_Shifted", "previous_output", "previous_output_time"]]
    targets = timestamps.rename(columns={"Timestamp": "Timestamp_Shifted"})[["Timestamp_Shifted"]].copy()
    frame = _attach_latest_features(targets, feature_events, output_events)
    return frame.dropna(subset=MODEL_FEATURES).reset_index(drop=True)


def _standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


def fit_ridge_regression(train_x: np.ndarray, train_y: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ train_y)


def predict_ridge(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return design @ coefficients


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - (np.sum(error**2) / denom)) if denom else 0.0
    within_025 = float(np.mean(np.abs(error) <= 0.25))
    within_050 = float(np.mean(np.abs(error) <= 0.50))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_0_25": within_025,
        "within_0_50": within_050,
    }


def train_quality_model(frame: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, dict, dict]:
    ordered = frame.sort_values("Timestamp_Shifted").reset_index(drop=True)
    split = max(10, int(len(ordered) * (1 - test_fraction)))
    train = ordered.iloc[:split].copy()
    holdout = ordered.iloc[split:].copy()

    train_x_raw = train[MODEL_FEATURES].to_numpy(dtype=float)
    holdout_x_raw = holdout[MODEL_FEATURES].to_numpy(dtype=float)
    train_x, holdout_x, mean, std = _standardize(train_x_raw, holdout_x_raw)
    coeffs = fit_ridge_regression(train_x, train["Output Parameter"].to_numpy(dtype=float))
    holdout["prediction"] = predict_ridge(holdout_x, coeffs).clip(0)
    metrics = regression_metrics(
        holdout["Output Parameter"].to_numpy(dtype=float),
        holdout["prediction"].to_numpy(dtype=float),
    )
    model = {
        "coefficients": coeffs,
        "mean": mean,
        "std": std,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "feature_names": MODEL_FEATURES,
    }
    return holdout, metrics, model


def predict_quality(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    result = frame.copy()
    x_raw = result[model["feature_names"]].to_numpy(dtype=float)
    x = (x_raw - model["mean"]) / model["std"]
    result["Prediction"] = predict_ridge(x, model["coefficients"]).clip(0)
    result["prediction_confidence"] = 0.72
    return result
