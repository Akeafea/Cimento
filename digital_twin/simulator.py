from __future__ import annotations

import pandas as pd


def simulate_control_effect(recommendations: pd.DataFrame) -> pd.DataFrame:
    simulated = recommendations.copy()
    saving = (
        simulated["fuel_rate_delta_pct"].abs() * 0.58
        + simulated["fan_speed_delta_pct"].abs() * 0.22
        + simulated["kiln_feed_delta_pct"].clip(lower=0) * 0.18
    ).clip(0, 6)
    risk_penalty = (1 - simulated["safety_score"]) * 1.6
    simulated["predicted_energy_saving_pct"] = (saving - risk_penalty).clip(0, 5).round(3)
    simulated["predicted_quality_shift"] = (
        simulated["kiln_feed_delta_pct"] * 0.10 - simulated["fuel_rate_delta_pct"].abs() * 0.025
    ).round(3)
    simulated["digital_twin_status"] = simulated["safety_score"].apply(
        lambda score: "safe_to_recommend" if score >= 0.72 else "operator_review_required"
    )
    return simulated
