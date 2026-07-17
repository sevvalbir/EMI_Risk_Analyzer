"""Shared constants, feature schema, and PCB input bounds."""

from __future__ import annotations

from typing import Any

FEATURE_COLUMNS: list[str] = [
    "Layer_Count",
    "Clock_Frequency_MHz",
    "Rise_Time_ns",
    "Signal_GND_Distance_mm",
    "Trace_Length_mm",
    "Stub_Length_mm",
    "Differential_Pair_Symmetry",
    "Ground_Plane",
    "Shielding",
    "Switching_Regulator",
]

FEATURE_LABELS: list[str] = [
    "Layer Count",
    "Clock Frequency (MHz)",
    "Rise Time (ns)",
    "Signal-GND Distance (mm)",
    "Trace Length (mm)",
    "Stub Length (mm)",
    "Differential Pair Symmetry (%)",
    "Ground Plane",
    "Shielding",
    "Switching Regulator",
]

PHYSICAL_REASONS: dict[str, str] = {
    "Layer Count": (
        "Lower layer count usually provides weaker return paths and increases EMI."
    ),
    "Clock Frequency (MHz)": (
        "Higher clock frequencies generate stronger harmonics, increasing EMI emissions."
    ),
    "Rise Time (ns)": (
        "Faster signal edges contain more high-frequency components, increasing EMI."
    ),
    "Signal-GND Distance (mm)": (
        "Increasing the distance enlarges the current loop area and radiation."
    ),
    "Trace Length (mm)": (
        "Long PCB traces behave like antennas and radiate more electromagnetic energy."
    ),
    "Stub Length (mm)": (
        "Long stubs create impedance discontinuities and signal reflections."
    ),
    "Differential Pair Symmetry (%)": (
        "Poor differential symmetry increases common-mode noise."
    ),
    "Ground Plane": (
        "A missing ground plane weakens return current paths and increases EMI."
    ),
    "Shielding": (
        "Without shielding, electromagnetic energy can radiate more easily."
    ),
    "Switching Regulator": (
        "Switching regulators generate high-frequency switching noise."
    ),
}

# Hard limits enforced in the UI / API (engineering + training OOD guardrails)
PCB_BOUNDS: dict[str, dict[str, Any]] = {
    "layer_count": {"allowed": (2, 4, 6, 8)},
    "clock_frequency": {"min": 1.0, "max": 2000.0, "warn_max": 600.0},
    "rise_time": {"min": 0.1, "max": 50.0},
    "signal_distance": {"min": 0.1, "max": 10.0},
    "trace_length": {"min": 1.0, "max": 500.0},
    "stub_length": {"min": 0.0, "max": 50.0},
    "differential_pair": {"min": 80.0, "max": 100.0},
}

IMPACT_HIGH = 0.10
IMPACT_MEDIUM = 0.03
