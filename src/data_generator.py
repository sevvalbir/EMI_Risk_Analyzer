"""Generate synthetic PCB EMI dataset (cwd-independent paths)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import DATA_DIR, DATASET_PATH, ensure_dir

np.random.seed(42)
num_samples = 1000


def calculate_emi_score(row: pd.Series) -> int:
    score = 0

    if row["Clock_Frequency_MHz"] > 300:
        score += 25
    if row["Rise_Time_ns"] < 2:
        score += 20
    if row["Signal_GND_Distance_mm"] > 1:
        score += 15
    if row["Trace_Length_mm"] > 150:
        score += 10
    if row["Stub_Length_mm"] > 5:
        score += 10
    if row["Differential_Pair_Symmetry"] < 95:
        score += 10
    if row["Ground_Plane"] == 0:
        score += 20
    if row["Shielding"] == 0:
        score += 10
    if row["Switching_Regulator"] == 1:
        score += 5

    layer = int(row["Layer_Count"])
    if layer == 2:
        score += 20
    elif layer == 4:
        score += 10
    elif layer == 6:
        score += 5

    return score


df = pd.DataFrame(
    {
        "PCB_ID": [f"PCB_{i + 1:04d}" for i in range(num_samples)],
        "Layer_Count": np.random.choice([2, 4, 6, 8], num_samples),
        "Clock_Frequency_MHz": np.random.randint(10, 600, num_samples),
        "Rise_Time_ns": np.round(np.random.uniform(0.5, 10, num_samples), 2),
        "Signal_GND_Distance_mm": np.round(
            np.random.uniform(0.1, 3.0, num_samples), 2
        ),
        "Trace_Length_mm": np.random.randint(20, 300, num_samples),
        "Stub_Length_mm": np.round(np.random.uniform(0, 10, num_samples), 2),
        "Differential_Pair_Symmetry": np.round(
            np.random.uniform(80, 100, num_samples), 1
        ),
        "Ground_Plane": np.random.choice([0, 1], num_samples),
        "Shielding": np.random.choice([0, 1], num_samples),
        "Switching_Regulator": np.random.choice([0, 1], num_samples),
    }
)

df["EMI_Score"] = df.apply(calculate_emi_score, axis=1)
df["Result"] = np.where(df["EMI_Score"] >= 60, "FAIL", "PASS")

print(df.head())

ensure_dir(DATA_DIR)
df.to_csv(DATASET_PATH, index=False)
print("\nDataset written to:", DATASET_PATH.resolve())
