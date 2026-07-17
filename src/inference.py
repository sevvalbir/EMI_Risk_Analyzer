"""
ML inference layer: prediction, SHAP explainability, recommendations.

Kept free of Streamlit so CLI scripts and the web UI share one pipeline.
"""

from __future__ import annotations

from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    IMPACT_HIGH,
    IMPACT_MEDIUM,
    PCB_BOUNDS,
    PHYSICAL_REASONS,
)
from src.paths import MODEL_PATH


class PCBValidationError(ValueError):
    """Raised when PCB parameters fail hard validation."""


def load_model(model_path=MODEL_PATH):
    """Load the trained classifier from disk."""
    path = MODEL_PATH if model_path is None else model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it with src/train_model.py first."
        )
    return joblib.load(path)


def build_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def validate_pcb_inputs(params: Mapping[str, Any]) -> list[str]:
    """
    Validate PCB parameters.

    Returns a list of non-fatal warning messages.
    Raises PCBValidationError for hard violations.
    """
    errors: list[str] = []
    warnings: list[str] = []

    allowed_layers = PCB_BOUNDS["layer_count"]["allowed"]
    layer = params.get("layer_count")
    if layer not in allowed_layers:
        errors.append(f"Layer Count must be one of {allowed_layers}, got {layer}.")

    numeric_keys = (
        "clock_frequency",
        "rise_time",
        "signal_distance",
        "trace_length",
        "stub_length",
        "differential_pair",
    )
    for key in numeric_keys:
        bounds = PCB_BOUNDS[key]
        try:
            value = float(params[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"Invalid or missing value for '{key}'.")
            continue
        if not np.isfinite(value):
            errors.append(f"'{key}' must be a finite number.")
            continue
        if value < bounds["min"] or value > bounds["max"]:
            errors.append(
                f"'{key}'={value} is outside allowed range "
                f"[{bounds['min']}, {bounds['max']}]."
            )

    clock = params.get("clock_frequency")
    warn_max = PCB_BOUNDS["clock_frequency"].get("warn_max")
    if (
        clock is not None
        and warn_max is not None
        and float(clock) > float(warn_max)
        and float(clock) <= PCB_BOUNDS["clock_frequency"]["max"]
    ):
        warnings.append(
            f"Clock frequency {clock} MHz exceeds the typical training range "
            f"(~{warn_max} MHz). Prediction may be less reliable (OOD)."
        )

    for flag in ("ground_plane", "shielding", "switching_regulator"):
        if flag not in params:
            errors.append(f"Missing boolean flag '{flag}'.")
        elif not isinstance(params[flag], (bool, np.bool_, int)):
            errors.append(f"'{flag}' must be a boolean.")

    if errors:
        raise PCBValidationError(" ".join(errors))
    return warnings


def pcb_to_dataframe(params: Mapping[str, Any]) -> pd.DataFrame:
    """Convert UI/CLI parameter dict to a one-row model feature frame."""
    validate_pcb_inputs(params)
    row = {
        "Layer_Count": int(params["layer_count"]),
        "Clock_Frequency_MHz": float(params["clock_frequency"]),
        "Rise_Time_ns": float(params["rise_time"]),
        "Signal_GND_Distance_mm": float(params["signal_distance"]),
        "Trace_Length_mm": float(params["trace_length"]),
        "Stub_Length_mm": float(params["stub_length"]),
        "Differential_Pair_Symmetry": float(params["differential_pair"]),
        "Ground_Plane": int(bool(params["ground_plane"])),
        "Shielding": int(bool(params["shielding"])),
        "Switching_Regulator": int(bool(params["switching_regulator"])),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def extract_shap_values(shap_values) -> np.ndarray:
    """Normalize SHAP output shapes across library / model versions."""
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
        values = shap_values[0, :, 1]
    else:
        values = shap_values[0]
    return np.asarray(values, dtype=float).reshape(-1)


def impact_label(impact: float) -> str:
    if impact >= IMPACT_HIGH:
        return "HIGH"
    if impact >= IMPACT_MEDIUM:
        return "MEDIUM"
    return "LOW"


def impact_color(impact: float) -> str:
    if impact >= IMPACT_HIGH:
        return "#E53935"
    if impact >= IMPACT_MEDIUM:
        return "#FB8C00"
    return "#43A047"


def risk_band(risk_score: float) -> tuple[str, str]:
    if risk_score >= 70:
        return "HIGH RISK", "error"
    if risk_score >= 40:
        return "MEDIUM RISK", "warning"
    return "LOW RISK", "success"


def build_recommendations(params: Mapping[str, Any]) -> list[dict[str, str]]:
    """Priority-ranked engineering recommendations from PCB thresholds."""
    recs: list[dict[str, str]] = []

    if float(params["clock_frequency"]) > 300:
        recs.append(
            {
                "priority": "High",
                "text": "Reduce clock frequency if system requirements allow.",
            }
        )
    if float(params["rise_time"]) < 2:
        recs.append(
            {
                "priority": "High",
                "text": "Increase rise time slightly to reduce high-frequency harmonics.",
            }
        )
    if not params["ground_plane"]:
        recs.append({"priority": "High", "text": "Add a continuous ground plane."})

    if float(params["signal_distance"]) > 1:
        recs.append(
            {
                "priority": "Medium",
                "text": "Reduce the distance between signal traces and ground plane.",
            }
        )
    if float(params["trace_length"]) > 150:
        recs.append(
            {"priority": "Medium", "text": "Shorten critical PCB traces."}
        )
    if float(params["stub_length"]) > 5:
        recs.append(
            {
                "priority": "Medium",
                "text": "Reduce stub length to minimize signal reflections.",
            }
        )

    if float(params["differential_pair"]) < 95:
        recs.append(
            {
                "priority": "Low",
                "text": "Improve differential pair routing symmetry.",
            }
        )
    if not params["shielding"]:
        recs.append(
            {
                "priority": "Low",
                "text": "Apply shielding around critical circuits.",
            }
        )
    if params["switching_regulator"]:
        recs.append(
            {
                "priority": "Low",
                "text": "Improve filtering around the switching regulator.",
            }
        )

    return recs


def build_input_summary(params: Mapping[str, Any]) -> pd.DataFrame:
    values = [
        params["layer_count"],
        params["clock_frequency"],
        params["rise_time"],
        params["signal_distance"],
        params["trace_length"],
        params["stub_length"],
        params["differential_pair"],
        "Available" if params["ground_plane"] else "Not Available",
        "Applied" if params["shielding"] else "Not Applied",
        "Present" if params["switching_regulator"] else "Not Present",
    ]
    return pd.DataFrame({"Parameter": FEATURE_LABELS, "Value": values})


def build_shap_dataframe(values: np.ndarray) -> pd.DataFrame:
    """Vectorized absolute SHAP impact table with engineering rationale."""
    impact = np.abs(np.asarray(values, dtype=float))
    if impact.shape[0] != len(FEATURE_LABELS):
        raise ValueError(
            f"SHAP value length {impact.shape[0]} != features {len(FEATURE_LABELS)}"
        )

    shap_df = pd.DataFrame(
        {
            "Parameter": FEATURE_LABELS,
            "Impact": impact,
        }
    )
    shap_df["Fiziksel Gerekçe (Why?)"] = shap_df["Parameter"].map(PHYSICAL_REASONS)
    shap_df["Risk Level"] = np.where(
        shap_df["Impact"] >= IMPACT_HIGH,
        "HIGH",
        np.where(shap_df["Impact"] >= IMPACT_MEDIUM, "MEDIUM", "LOW"),
    )
    shap_df = shap_df.sort_values("Impact", ascending=False).reset_index(drop=True)
    shap_df["Impact"] = shap_df["Impact"].round(4)
    return shap_df


def predict_emi(model, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run classifier prediction and return serializable metrics."""
    input_data = pcb_to_dataframe(params)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    classes = list(model.classes_)
    fail_idx = classes.index("FAIL") if "FAIL" in classes else 0
    risk_score = float(probability[fail_idx]) * 100.0
    confidence = float(np.max(probability) * 100.0)
    band_label, band_severity = risk_band(risk_score)
    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "risk_score": risk_score,
        "band_label": band_label,
        "band_severity": band_severity,
        "input_data": input_data,
    }


def explain_emi(explainer, input_data: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP impacts for a one-row feature frame."""
    shap_values = explainer.shap_values(input_data)
    values = extract_shap_values(shap_values)
    return build_shap_dataframe(values)


def analyze_pcb(model, explainer, params: Mapping[str, Any]) -> dict[str, Any]:
    """
    Full analysis pipeline used by Streamlit and CLI.

    Returns a JSON-serializable-ish dict (DataFrames kept as objects for UI).
    """
    warnings = validate_pcb_inputs(params)
    pred = predict_emi(model, params)
    shap_df = explain_emi(explainer, pred["input_data"])
    recommendations = build_recommendations(params)
    input_summary = build_input_summary(params)
    high_count = sum(1 for r in recommendations if r["priority"] == "High")

    return {
        "warnings": warnings,
        "prediction": pred["prediction"],
        "confidence": pred["confidence"],
        "risk_score": pred["risk_score"],
        "band_label": pred["band_label"],
        "band_severity": pred["band_severity"],
        "input_summary": input_summary,
        "shap_df": shap_df,
        "recommendations": recommendations,
        "high_count": high_count,
    }
