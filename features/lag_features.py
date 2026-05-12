from __future__ import annotations

import numpy as np
import pandas as pd

from project_config import ENERGY_WEIGHTS, PROCESS_FEATURES


def _minmax(series: pd.Series) -> pd.Series:
    spread = series.max() - series.min()
    if spread == 0:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / spread


def add_process_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    scaled = pd.DataFrame({col: _minmax(featured[col]) for col in PROCESS_FEATURES})

    featured["Estimated_Energy"] = sum(
        ENERGY_WEIGHTS[col] * scaled[col] for col in PROCESS_FEATURES
    ) * 100
    featured["Specific_Energy_kwh_t"] = 82 + (featured["Estimated_Energy"] * 0.42)
    featured["Specific_Heat_kcal_kg"] = 720 + (scaled["Temp_2"] * 85) + (scaled["Process_Load"] * 28)
    featured["Clinker_Quality_Index"] = (
        96
        - (scaled["Temp_2"].sub(0.55).abs() * 9)
        - (scaled["Process_Load"].sub(0.50).abs() * 7)
        - (scaled["Flow_Pressure"].sub(0.45).abs() * 4)
    ).clip(70, 99)

    group = featured.groupby("unit", group_keys=False)
    for lag in (5, 15, 30):
        featured[f"Energy_lag_{lag}"] = group["Estimated_Energy"].shift(lag)
        featured[f"Temp_2_lag_{lag}"] = group["Temp_2"].shift(lag)

    for window in (15, 30):
        featured[f"Energy_roll_mean_{window}"] = group["Estimated_Energy"].transform(
            lambda s: s.rolling(window, min_periods=3).mean()
        )
        featured[f"Energy_roll_std_{window}"] = group["Estimated_Energy"].transform(
            lambda s: s.rolling(window, min_periods=3).std()
        )
        featured[f"Temp_2_roll_std_{window}"] = group["Temp_2"].transform(
            lambda s: s.rolling(window, min_periods=3).std()
        )

    feature_cols = [c for c in featured.columns if "lag_" in c or "roll_" in c]
    featured[feature_cols] = group[feature_cols].transform(lambda part: part.bfill().ffill())
    featured[feature_cols] = featured[feature_cols].fillna(featured[feature_cols].median(numeric_only=True))
    featured.replace([np.inf, -np.inf], np.nan, inplace=True)
    return featured.ffill().bfill()
