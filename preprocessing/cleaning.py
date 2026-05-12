from __future__ import annotations

import numpy as np
import pandas as pd

from project_config import PROCESS_FEATURES


def clean_process_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.sort_values(["unit", "cycle"]).reset_index(drop=True)
    missing_before = cleaned[PROCESS_FEATURES].isna().sum(axis=1)

    cleaned[PROCESS_FEATURES] = (
        cleaned.groupby("unit")[PROCESS_FEATURES]
        .transform(lambda part: part.interpolate(limit_direction="both"))
        .ffill()
        .bfill()
    )

    clipped_columns = []
    for col in PROCESS_FEATURES:
        low = cleaned[col].quantile(0.005)
        high = cleaned[col].quantile(0.995)
        clipped = ~cleaned[col].between(low, high)
        clipped_columns.append(clipped.astype(int))
        cleaned[col] = cleaned[col].clip(low, high)

    clipped_count = np.sum(clipped_columns, axis=0) if clipped_columns else 0
    penalty = (missing_before * 0.08) + (clipped_count * 0.04)
    cleaned["sensor_reliability_score"] = (1.0 - penalty).clip(0.55, 1.0).round(4)
    cleaned["cleaning_status"] = np.where(penalty > 0, "corrected", "ok")
    return cleaned
