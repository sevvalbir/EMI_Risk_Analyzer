"""CLI SHAP explanation for a sample PCB (cwd-independent)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FEATURE_COLUMNS
from src.inference import (
    build_explainer,
    build_recommendations,
    build_shap_dataframe,
    extract_shap_values,
    load_model,
)
from src.paths import DATASET_PATH, MODEL_PATH, REPORTS_DIR, ensure_dir

model = load_model(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)
X = df[FEATURE_COLUMNS]

explainer = build_explainer(model)
shap_values = explainer.shap_values(X)

sample = X.iloc[[0]]
pcb = sample.iloc[0]

params = {
    "layer_count": int(pcb["Layer_Count"]),
    "clock_frequency": float(pcb["Clock_Frequency_MHz"]),
    "rise_time": float(pcb["Rise_Time_ns"]),
    "signal_distance": float(pcb["Signal_GND_Distance_mm"]),
    "trace_length": float(pcb["Trace_Length_mm"]),
    "stub_length": float(pcb["Stub_Length_mm"]),
    "differential_pair": float(pcb["Differential_Pair_Symmetry"]),
    "ground_plane": bool(pcb["Ground_Plane"]),
    "shielding": bool(pcb["Shielding"]),
    "switching_regulator": bool(pcb["Switching_Regulator"]),
}

print("\n==============================")
print(" PCB ANALYSIS")
print("==============================")
print(f"Layer Count                 : {params['layer_count']} Layers")
print(f"Clock Frequency             : {params['clock_frequency']:.0f} MHz")
print(f"Rise Time                   : {params['rise_time']:.2f} ns")
print(f"Signal-GND Distance         : {params['signal_distance']:.2f} mm")
print(f"Trace Length                : {params['trace_length']:.0f} mm")
print(f"Stub Length                 : {params['stub_length']:.2f} mm")
print(f"Differential Pair Symmetry  : {params['differential_pair']:.1f} %")
print(
    "Ground Plane                : "
    f"{'Available' if params['ground_plane'] else 'Not Available'}"
)
print(
    "Shielding                   : "
    f"{'Applied' if params['shielding'] else 'Not Applied'}"
)
print(
    "Switching Regulator         : "
    f"{'Present' if params['switching_regulator'] else 'Not Present'}"
)
print("==============================")

recommendations = build_recommendations(params)

# Sample-row SHAP (compatible across shap versions)
sample_shap = explainer.shap_values(sample)
values = extract_shap_values(sample_shap)
importance = build_shap_dataframe(values)

print("\n==============================")
print(" EMI RISK FACTORS (ALL PARAMETERS)")
print("==============================")

for i, row in enumerate(importance.to_dict(orient="records"), start=1):
    print(f"\n{i}. {row['Parameter']}")
    print(f"Impact Score : {row['Impact']:.4f}")
    print(f"Risk Level  : {row['Risk Level']}")
    print("Reason:")
    print(row["Fiziksel Gerekçe (Why?)"])
    print("-" * 50)

plt.figure(figsize=(10, 6))
plt.barh(importance["Parameter"], importance["Impact"])
plt.xlabel("SHAP Impact")
plt.ylabel("PCB Parameters")
plt.title("EMI Risk Factor Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

ensure_dir(REPORTS_DIR)
out_png = REPORTS_DIR / "emi_risk_factors.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print("\nChart saved:", out_png.resolve())

print("\n==============================")
print(" EMI DESIGN RECOMMENDATIONS")
print("==============================")
if recommendations:
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. [{rec['priority']}] {rec['text']}")
else:
    print("No major EMI improvements are required.")
print("==============================")

plt.show()
