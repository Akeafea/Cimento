from __future__ import annotations

import numpy as np
import pandas as pd


def _robust_z(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    scale = 1.4826 * mad if mad else series.std() or 1.0
    return (series - median) / scale


def score_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    scored = df.copy()
    components = pd.DataFrame(
        {
            "energy": _robust_z(scored["Estimated_Energy"]).abs(),
            "energy_volatility": _robust_z(scored["Energy_roll_std_30"]).abs(),
            "temp_volatility": _robust_z(scored["Temp_2_roll_std_30"]).abs(),
            "quality": _robust_z(100 - scored["Clinker_Quality_Index"]).abs(),
        }
    )
    raw = (
        0.34 * components["energy"]
        + 0.27 * components["energy_volatility"]
        + 0.22 * components["temp_volatility"]
        + 0.17 * components["quality"]
    )
    scored["anomaly_score"] = (1 / (1 + np.exp(-(raw - 2.0)))).clip(0, 1)
    threshold = scored["anomaly_score"].quantile(1 - contamination)
    scored["if_anomaly"] = (scored["anomaly_score"] >= threshold).astype(int)
    distance = (scored["anomaly_score"] - threshold).abs()
    scored["anomaly_confidence"] = (0.55 + (distance / (distance.max() or 1)) * 0.45).clip(0.55, 0.99)
    scored["if_threshold"] = threshold
    return scored
