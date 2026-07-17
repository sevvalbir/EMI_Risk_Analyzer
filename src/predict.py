"""CLI EMI prediction using the shared inference layer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import PCBValidationError, load_model, predict_emi
from src.paths import MODEL_PATH

print("==============================")
print(" EMI RISK ANALYZER")
print("==============================")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

try:
    layer_count = int(input("Layer Count (2/4/6/8): "))
    clock_frequency = float(input("Clock Frequency (MHz): "))
    rise_time = float(input("Rise Time (ns): "))
    signal_gnd_distance = float(input("Signal-GND Distance (mm): "))
    trace_length = float(input("Trace Length (mm): "))
    stub_length = float(input("Stub Length (mm): "))
    differential_symmetry = float(input("Differential Pair Symmetry (%): "))
    ground_plane = int(input("Ground Plane (0=No, 1=Yes): "))
    shielding = int(input("Shielding (0=No, 1=Yes): "))
    switching_regulator = int(input("Switching Regulator (0=No, 1=Yes): "))
except ValueError:
    print("ERROR: Invalid numeric input.")
    sys.exit(1)

params = {
    "layer_count": layer_count,
    "clock_frequency": clock_frequency,
    "rise_time": rise_time,
    "signal_distance": signal_gnd_distance,
    "trace_length": trace_length,
    "stub_length": stub_length,
    "differential_pair": differential_symmetry,
    "ground_plane": bool(ground_plane),
    "shielding": bool(shielding),
    "switching_regulator": bool(switching_regulator),
}

try:
    result = predict_emi(model, params)
except PCBValidationError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

print("\n==============================")
print(" RESULT")
print("==============================")
print("EMI Risk:", result["prediction"])
print(f"FAIL Probability: {result['risk_score']:.2f}%")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Risk Band: {result['band_label']}")
