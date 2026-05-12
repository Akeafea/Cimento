from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(value: pd.Series, low: float, high: float) -> pd.Series:
    return value.clip(lower=low, upper=high)


def build_mpc_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1).copy()
    energy_gap = (latest["Estimated_Energy"] - latest["Estimated_Energy"].median()) / (
        latest["Estimated_Energy"].std() or 1.0
    )
    risk = latest["ensemble_risk_score"]

    rec = pd.DataFrame(
        {
            "unit": latest["unit"].astype(int),
            "cycle": latest["cycle"].astype(int),
            "mode": "open_loop_operator_approval",
            "fuel_rate_delta_pct": _clip(-(energy_gap * 0.85 + risk * 3.2), -5.0, 1.0),
            "fan_speed_delta_pct": _clip(-(latest["Temp_2_roll_std_30"] * 1.15), -3.0, 2.0),
            "kiln_feed_delta_pct": np.where(risk > 0.68, -1.2, 0.8),
        }
    )
    rec["safety_score"] = (
        0.42 * latest["sensor_reliability_score"].to_numpy()
        + 0.33 * (1 - risk.to_numpy())
        + 0.25 * latest["sequence_confidence"].to_numpy()
    ).clip(0, 1)
    rec["recommendation_confidence"] = (
        0.50 * latest["anomaly_confidence"].to_numpy()
        + 0.30 * latest["sequence_confidence"].to_numpy()
        + 0.20 * latest["sensor_reliability_score"].to_numpy()
    ).clip(0, 0.99)
    rec["operator_message"] = np.where(
        rec["safety_score"] >= 0.72,
        "Enerji dusurme onerisi guvenli sinirlar icinde; operator onayi bekleniyor.",
        "Risk yuksek; once proses stabilitesi ve sensor guvenilirligi kontrol edilmeli.",
    )
    return rec.round(4)
