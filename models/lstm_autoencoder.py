from __future__ import annotations

import numpy as np
import pandas as pd

from project_config import PROCESS_FEATURES


def _scaled_abs_deviation(series: pd.Series, window: int = 30) -> pd.Series:
    rolling = series.rolling(window, min_periods=5).median().bfill().ffill()
    deviation = (series - rolling).abs()
    high = deviation.quantile(0.995) or deviation.max() or 1.0
    return (deviation / high).clip(0, 1)


def score_sequence_risk(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    deviations = []
    for col in PROCESS_FEATURES + ["Estimated_Energy", "Specific_Heat_kcal_kg"]:
        deviations.append(
            scored.groupby("unit", group_keys=False)[col].apply(_scaled_abs_deviation)
        )

    reconstruction = np.mean(deviations, axis=0)
    trend = (
        scored.groupby("unit")["Estimated_Energy"]
        .diff(15)
        .abs()
        .fillna(0)
        .pipe(lambda s: s / (s.quantile(0.995) or s.max() or 1.0))
        .clip(0, 1)
    )
    scored["reconstruction_error"] = (0.72 * reconstruction + 0.28 * trend).clip(0, 1)
    threshold = scored["reconstruction_error"].quantile(0.98)
    scored["lstm_anomaly"] = (scored["reconstruction_error"] >= threshold).astype(int)
    scored["quality_risk_60m"] = (
        0.52 * scored["reconstruction_error"]
        + 0.28 * (100 - scored["Clinker_Quality_Index"]) / 30
        + 0.20 * scored["Energy_roll_std_30"].fillna(0) / (scored["Energy_roll_std_30"].quantile(0.995) or 1)
    ).clip(0, 1)
    scored["sequence_confidence"] = (0.60 + scored["quality_risk_60m"] * 0.35).clip(0.60, 0.97)
    scored["lstm_threshold"] = threshold
    return scored
