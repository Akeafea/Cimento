from __future__ import annotations

from pathlib import Path

import pandas as pd

from project_config import ROOT

CAX_ROOT = ROOT / "CAX data"
TRAIN_QUALITY_FILE = CAX_ROOT / "CAX_Train_Quality (1)" / "CAX_Train_Quality.csv"
TEST_QUALITY_FILE = CAX_ROOT / "CAX_Test_Quality" / "CAX_Test_Quality.csv"
SUBMISSION_FILE = CAX_ROOT / "CAX_Freelime_Submission_File.csv"


def cax_files_available() -> bool:
    return TRAIN_QUALITY_FILE.exists() and SUBMISSION_FILE.exists()


def read_cax_quality(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp_Shifted"] = pd.to_datetime(df["Timestamp_Shifted"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df.dropna(subset=["Timestamp_Shifted", "Parameter", "Value"])


def read_submission_timestamps(path: Path = SUBMISSION_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df.dropna(subset=["Timestamp"])
