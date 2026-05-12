from __future__ import annotations

import pandas as pd

from project_config import DATA_FILE, RAW_COLUMNS, SENSOR_MAP


def load_scada_like_data(path=DATA_FILE, max_units: int | None = None) -> pd.DataFrame:
    """Load NASA turbofan data as a deterministic SCADA-like cement process feed."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
    if max_units:
        selected_units = sorted(df["unit"].unique())[:max_units]
        df = df[df["unit"].isin(selected_units)].copy()

    selected = ["unit", "cycle", "op1", "op2", "op3"] + list(SENSOR_MAP)
    df = df[selected].rename(columns=SENSOR_MAP)
    df["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df["cycle"], unit="min")
    df["source"] = "mock_scada_csv"
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1)
