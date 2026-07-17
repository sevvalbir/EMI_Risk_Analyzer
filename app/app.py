# -*- coding: utf-8 -*-
"""EMI Risk Analyzer — bilingual industrial Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PCB_BOUNDS
from src.inference import (
    PCBValidationError,
    analyze_pcb,
    build_explainer,
    impact_color,
    load_model,
    risk_band,
    validate_pcb_inputs,
)
from src.pdf_report import generate_emi_pdf_report
from src.paths import DATASET_PATH, MODEL_PATH

# ---------------------------------------------------------------------------
# Translations (UTF-8). Resolve with t("key").
# ---------------------------------------------------------------------------
translations = {
    "en": {
        "app_title": "EMI Risk Analyzer",
        "header_subtitle": (
            "PCB Electromagnetic Interference Prediction & Diagnostics"
        ),
        "sidebar_caption": "PCB design parameters",
        "language": "Language",
        "section_layer": "Layer Parameters",
        "section_signal": "Signal Parameters",
        "section_mitigation": "Mitigation Features",
        "layer_count": "Layer Count",
        "clock_frequency": "Clock Frequency (MHz)",
        "rise_time": "Rise Time (ns)",
        "signal_distance": "Signal-GND Distance (mm)",
        "trace_length": "Trace Length (mm)",
        "stub_length": "Stub Length (mm)",
        "differential_pair": "Differential Pair Symmetry (%)",
        "ground_plane": "Ground Plane",
        "shielding": "Shielding",
        "switching_regulator": "Switching Regulator",
        "help_clock": "Higher clocks produce stronger harmonics.",
        "help_rise": "Faster edges increase high-frequency EMI content.",
        "analyze": "Analyze PCB",
        "dashboard_title": "EMI Risk Dashboard",
        "dashboard_caption": (
            "PCB EMI risk prediction, SHAP explainability, and design recommendations"
        ),
        "idle_info": (
            "Configure PCB parameters in the control panel, then click Analyze PCB."
        ),
        "stale_warning": (
            "Control-panel parameters changed since the last analysis. "
            "Click Analyze PCB again to refresh results."
        ),
        "control_panel": "Control Panel",
        "invalid_params": "Invalid PCB parameters",
        "validation_failed": "Validation failed",
        "analysis_failed": "Analysis failed",
        "model_load_failed": "Failed to load model resources",
        "pdf_failed": "PDF report could not be generated",
        "ood_clock": (
            "Clock frequency exceeds the typical training range (~{warn_max} MHz). "
            "Prediction may be less reliable (out of distribution)."
        ),
        "fail_title": "FAIL — EMI Risk Detected",
        "fail_body": (
            "This PCB configuration is predicted to fail EMI compliance. "
            "Risk band: {band} ({score:.1f}/100). "
            "Review root causes and apply high-priority hardware fixes before layout freeze."
        ),
        "pass_title": "PASS — Within Acceptable EMI Limits",
        "pass_body": (
            "This PCB configuration is predicted to pass EMI screening. "
            "Risk band: {band} ({score:.1f}/100). "
            "Maintain ground plane continuity, shielding, and controlled edge rates."
        ),
        "pass_residual": (
            "PASS overall, but residual EMI risk remains elevated. "
            "Review medium and low priority fixes before production."
        ),
        "certainty_high": "High model certainty ({c:.1f}%). Treat this FAIL as actionable.",
        "certainty_mod_fail": (
            "Moderate certainty ({c:.1f}%). Validate with laboratory EMC testing."
        ),
        "certainty_mod_pass": (
            "Lower model certainty ({c:.1f}%). Confirm with measurement."
        ),
        "metric_prediction": "Prediction",
        "metric_risk": "EMI Risk Score",
        "metric_confidence": "Model Confidence",
        "metric_fixes": "High-Priority Fixes",
        "delta_fail": "EMI FAIL",
        "delta_pass": "EMI PASS",
        "delta_action": "Action needed",
        "delta_none": "None",
        "band_high": "HIGH RISK",
        "band_medium": "MEDIUM RISK",
        "band_low": "LOW RISK",
        "risk_progress": "EMI Risk Score · FAIL probability = {score:.1f}/100",
        "download_pdf": "Download PDF Report",
        "tab_pcb": "PCB Analysis",
        "tab_shap": "SHAP Risk Factors",
        "tab_recs": "Design Recommendations",
        "pcb_inputs": "Current PCB Inputs",
        "eng_notes": "Parameter engineering notes",
        "shap_ranking": "Feature Impact Ranking",
        "shap_chart": "SHAP importance chart",
        "shap_xlabel": "Absolute SHAP Impact",
        "shap_title": "EMI Risk Factor Importance",
        "col_parameter": "Parameter",
        "col_impact": "Impact",
        "col_level": "Risk Level",
        "col_reason": "Engineering Rationale",
        "col_value": "Value",
        "mitigation": "Mitigation Actions",
        "no_recs": "No major EMI improvements are required for this design.",
        "priority_high": "High Priority",
        "priority_medium": "Medium Priority",
        "priority_low": "Low Priority",
        "no_high": "No high-priority issues.",
        "no_medium": "No medium-priority issues.",
        "no_low": "No low-priority issues.",
        "rcf_title": "Risk, Cause, and Engineering Fix",
        "rcf_intro": (
            "Potential EMI risk, SHAP-backed root cause, and hardware corrective action."
        ),
        "rcf_risk": "Potential Risk",
        "rcf_cause": "Root Cause",
        "rcf_fix": "How to Fix",
        "rcf_empty": "No critical risk drivers identified for the current design.",
        "risk_fail": "High EMI compliance risk (model predicts FAIL)",
        "risk_pass_elevated": "Elevated residual EMI risk (PASS with elevated score)",
        "risk_pass_low": "Low EMI risk (model predicts PASS)",
        "available": "Available",
        "not_available": "Not Available",
        "applied": "Applied",
        "not_applied": "Not Applied",
        "present": "Present",
        "not_present": "Not Present",
        "level_high": "HIGH",
        "level_medium": "MEDIUM",
        "level_low": "LOW",
        "feat_Layer Count": "Layer Count",
        "feat_Clock Frequency (MHz)": "Clock Frequency (MHz)",
        "feat_Rise Time (ns)": "Rise Time (ns)",
        "feat_Signal-GND Distance (mm)": "Signal-GND Distance (mm)",
        "feat_Trace Length (mm)": "Trace Length (mm)",
        "feat_Stub Length (mm)": "Stub Length (mm)",
        "feat_Differential Pair Symmetry (%)": "Differential Pair Symmetry (%)",
        "feat_Ground Plane": "Ground Plane",
        "feat_Shielding": "Shielding",
        "feat_Switching Regulator": "Switching Regulator",
        "why_Layer Count": (
            "Lower layer count usually provides weaker return paths and increases EMI."
        ),
        "why_Clock Frequency (MHz)": (
            "Higher clock frequencies generate stronger harmonics, increasing EMI emissions."
        ),
        "why_Rise Time (ns)": (
            "Faster signal edges contain more high-frequency components, increasing EMI."
        ),
        "why_Signal-GND Distance (mm)": (
            "Increasing the distance enlarges the current loop area and radiation."
        ),
        "why_Trace Length (mm)": (
            "Long PCB traces behave like antennas and radiate more electromagnetic energy."
        ),
        "why_Stub Length (mm)": (
            "Long stubs create impedance discontinuities and signal reflections."
        ),
        "why_Differential Pair Symmetry (%)": (
            "Poor differential symmetry increases common-mode noise."
        ),
        "why_Ground Plane": (
            "A missing ground plane weakens return current paths and increases EMI."
        ),
        "why_Shielding": (
            "Without shielding, electromagnetic energy can radiate more easily."
        ),
        "why_Switching Regulator": (
            "Switching regulators generate high-frequency switching noise."
        ),
        "rec_clock": "Reduce clock frequency if system requirements allow.",
        "rec_rise": "Increase rise time slightly to reduce high-frequency harmonics.",
        "rec_ground": "Add a continuous ground plane.",
        "rec_distance": "Reduce the distance between signal traces and ground plane.",
        "rec_trace": "Shorten critical PCB traces.",
        "rec_stub": "Reduce stub length to minimize signal reflections.",
        "rec_diff": "Improve differential pair routing symmetry.",
        "rec_shield": "Apply shielding around critical circuits.",
        "rec_switch": "Improve filtering around the switching regulator.",
        "cause_prefix": "Primary SHAP driver: {feature} (impact {impact:.4f}).",
        "cause_value": "Current value: {value}.",
        "loading_model": "Loading EMI model...",
        "running_analysis": "Calculating model predictions...",
        "building_pdf": "Generating engineering PDF report...",
    },
    "tr": {
        "app_title": "EMI Risk Analizörü",
        "header_subtitle": (
            "PCB Elektromanyetik Girişim Tahmini ve Teşhisi"
        ),
        "sidebar_caption": "PCB tasarım parametreleri",
        "language": "Dil",
        "section_layer": "Katman Parametreleri",
        "section_signal": "Sinyal Parametreleri",
        "section_mitigation": "Azaltma Önlemleri",
        "layer_count": "Katman Sayısı",
        "clock_frequency": "Saat Frekansı (MHz)",
        "rise_time": "Yükselme Süresi (ns)",
        "signal_distance": "Sinyal-GND Mesafesi (mm)",
        "trace_length": "İz Uzunluğu (mm)",
        "stub_length": "Stub Uzunluğu (mm)",
        "differential_pair": "Diferansiyel Çift Simetrisi (%)",
        "ground_plane": "Toprak Düzlemi",
        "shielding": "Ekranlama",
        "switching_regulator": "Anahtarlamalı Regülatör",
        "help_clock": "Yüksek saat frekansları daha güçlü harmonikler üretir.",
        "help_rise": "Daha hızlı kenarlar yüksek frekanslı EMI içeriğini artırır.",
        "analyze": "PCB Analiz Et",
        "dashboard_title": "EMI Risk Panosu",
        "dashboard_caption": (
            "PCB EMI risk tahmini, SHAP açıklanabilirliği ve tasarım önerileri"
        ),
        "idle_info": (
            "PCB parametrelerini kontrol panelinden ayarlayın, ardından PCB Analiz Et "
            "düğmesine basın."
        ),
        "stale_warning": (
            "Kontrol paneli parametreleri son analizden sonra değişti. "
            "Sonuçları yenilemek için tekrar PCB Analiz Et düğmesine basın."
        ),
        "control_panel": "Kontrol Paneli",
        "invalid_params": "Geçersiz PCB parametreleri",
        "validation_failed": "Doğrulama başarısız",
        "analysis_failed": "Analiz başarısız",
        "model_load_failed": "Model kaynakları yüklenemedi",
        "pdf_failed": "PDF raporu oluşturulamadı",
        "ood_clock": (
            "Saat frekansı tipik eğitim aralığının üzerinde (~{warn_max} MHz). "
            "Tahmin daha az güvenilir olabilir (dağılım dışı)."
        ),
        "fail_title": "FAIL — EMI Riski Tespit Edildi",
        "fail_body": (
            "Bu PCB konfigürasyonunun EMI uygunluğunu geçemeyeceği tahmin edildi. "
            "Risk bandı: {band} ({score:.1f}/100). "
            "Kök nedenleri inceleyin ve yerleşim dondurulmadan önce yüksek öncelikli "
            "donanım düzeltmelerini uygulayın."
        ),
        "pass_title": "PASS — Kabul Edilebilir EMI Sınırları İçinde",
        "pass_body": (
            "Bu PCB konfigürasyonunun EMI tarama sınavını geçeceği tahmin edildi. "
            "Risk bandı: {band} ({score:.1f}/100). "
            "Toprak düzlemi sürekliliği, ekranlama ve kontrollü kenar hızlarını koruyun."
        ),
        "pass_residual": (
            "Genel sonuç PASS olsa da artık EMI riski yüksek kalmıştır. "
            "Üretime geçmeden önce orta ve düşük öncelikli düzeltmeleri gözden geçirin."
        ),
        "certainty_high": (
            "Yüksek model güveni ({c:.1f}%). Bu FAIL sonucunu uygulanabilir kabul edin."
        ),
        "certainty_mod_fail": (
            "Orta güven ({c:.1f}%). Laboratuvar EMC testi ile doğrulayın."
        ),
        "certainty_mod_pass": (
            "Daha düşük model güveni ({c:.1f}%). Ölçüm ile teyit edin."
        ),
        "metric_prediction": "Tahmin",
        "metric_risk": "EMI Risk Skoru",
        "metric_confidence": "Model Güveni",
        "metric_fixes": "Yüksek Öncelikli Düzeltme",
        "delta_fail": "EMI FAIL",
        "delta_pass": "EMI PASS",
        "delta_action": "Müdahale gerekli",
        "delta_none": "Yok",
        "band_high": "YÜKSEK RİSK",
        "band_medium": "ORTA RİSK",
        "band_low": "DÜŞÜK RİSK",
        "risk_progress": "EMI Risk Skoru · FAIL olasılığı = {score:.1f}/100",
        "download_pdf": "PDF Raporu İndir",
        "tab_pcb": "PCB Analizi",
        "tab_shap": "SHAP Risk Faktörleri",
        "tab_recs": "Tasarım Önerileri",
        "pcb_inputs": "Mevcut PCB Girdisi",
        "eng_notes": "Parametre mühendislik notları",
        "shap_ranking": "Özellik Etki Sıralaması",
        "shap_chart": "SHAP önem grafiği",
        "shap_xlabel": "Mutlak SHAP Etkisi",
        "shap_title": "EMI Risk Faktör Önemi",
        "col_parameter": "Parametre",
        "col_impact": "Etki",
        "col_level": "Risk Seviyesi",
        "col_reason": "Mühendislik Gerekçesi",
        "col_value": "Değer",
        "mitigation": "Azaltma Aksiyonları",
        "no_recs": "Bu tasarım için büyük EMI iyileştirmesi gerekmiyor.",
        "priority_high": "Yüksek Öncelik",
        "priority_medium": "Orta Öncelik",
        "priority_low": "Düşük Öncelik",
        "no_high": "Yüksek öncelikli sorun yok.",
        "no_medium": "Orta öncelikli sorun yok.",
        "no_low": "Düşük öncelikli sorun yok.",
        "rcf_title": "Risk, Neden ve Mühendislik Çözümü",
        "rcf_intro": (
            "Olası EMI riski, SHAP destekli kök neden ve donanım seviyesinde düzeltici müdahale."
        ),
        "rcf_risk": "Olası Risk",
        "rcf_cause": "Riskin Sebebi",
        "rcf_fix": "Çözüm Önerisi",
        "rcf_empty": "Mevcut tasarım için kritik risk sürücüsü tespit edilmedi.",
        "risk_fail": "Yüksek EMI uygunluk riski (model FAIL tahmin ediyor)",
        "risk_pass_elevated": "Yüksek artık EMI riski (PASS ancak skor yüksek)",
        "risk_pass_low": "Düşük EMI riski (model PASS tahmin ediyor)",
        "available": "Var",
        "not_available": "Yok",
        "applied": "Uygulanmış",
        "not_applied": "Uygulanmamış",
        "present": "Mevcut",
        "not_present": "Mevcut değil",
        "level_high": "YÜKSEK",
        "level_medium": "ORTA",
        "level_low": "DÜŞÜK",
        "feat_Layer Count": "Katman Sayısı",
        "feat_Clock Frequency (MHz)": "Saat Frekansı (MHz)",
        "feat_Rise Time (ns)": "Yükselme Süresi (ns)",
        "feat_Signal-GND Distance (mm)": "Sinyal-GND Mesafesi (mm)",
        "feat_Trace Length (mm)": "İz Uzunluğu (mm)",
        "feat_Stub Length (mm)": "Stub Uzunluğu (mm)",
        "feat_Differential Pair Symmetry (%)": "Diferansiyel Çift Simetrisi (%)",
        "feat_Ground Plane": "Toprak Düzlemi",
        "feat_Shielding": "Ekranlama",
        "feat_Switching Regulator": "Anahtarlamalı Regülatör",
        "why_Layer Count": (
            "Düşük katman sayısı genelde zayıf dönüş yolları sağlar ve EMI'yi artırır."
        ),
        "why_Clock Frequency (MHz)": (
            "Yüksek saat frekansları daha güçlü harmonikler üreterek EMI emisyonunu artırır."
        ),
        "why_Rise Time (ns)": (
            "Daha hızlı sinyal kenarları daha fazla yüksek frekans bileşeni içerir ve EMI'yi artırır."
        ),
        "why_Signal-GND Distance (mm)": (
            "Mesafenin artması akım döngü alanını genişletir ve ışınım artar."
        ),
        "why_Trace Length (mm)": (
            "Uzun PCB izleri anten gibi davranır ve daha fazla elektromanyetik enerji yayar."
        ),
        "why_Stub Length (mm)": (
            "Uzun stub'lar empedans süreksizlikleri ve sinyal yansımaları oluşturur."
        ),
        "why_Differential Pair Symmetry (%)": (
            "Zayıf diferansiyel simetri ortak mod gürültüsünü artırır."
        ),
        "why_Ground Plane": (
            "Eksik toprak düzlemi dönüş akım yollarını zayıflatır ve EMI'yi artırır."
        ),
        "why_Shielding": (
            "Ekranlama olmadan elektromanyetik enerji daha kolay yayılabilir."
        ),
        "why_Switching Regulator": (
            "Anahtarlamalı regülatörler yüksek frekanslı anahtarlama gürültüsü üretir."
        ),
        "rec_clock": (
            "Sistem gereksinimleri izin veriyorsa saat frekansını düşürün."
        ),
        "rec_rise": (
            "Yüksek frekanslı harmonikleri azaltmak için yükselme süresini hafifçe artırın."
        ),
        "rec_ground": "Sürekli bir toprak düzlemi ekleyin.",
        "rec_distance": (
            "Sinyal izleri ile toprak düzlemi arasındaki mesafeyi azaltın."
        ),
        "rec_trace": "Kritik PCB izlerini kısaltın.",
        "rec_stub": (
            "Sinyal yansımalarını azaltmak için stub uzunluğunu düşürün."
        ),
        "rec_diff": "Diferansiyel çift yönlendirme simetrisini iyileştirin.",
        "rec_shield": "Kritik devrelerin etrafına ekranlama uygulayın.",
        "rec_switch": (
            "Anahtarlamalı regülatör etrafında filtrelemeyi iyileştirin."
        ),
        "cause_prefix": "Birincil SHAP sürücüsü: {feature} (etki {impact:.4f}).",
        "cause_value": "Mevcut değer: {value}.",
        "loading_model": "EMI modeli yükleniyor...",
        "running_analysis": "Model sonuçları hesaplanıyor...",
        "building_pdf": "Mühendislik PDF raporu oluşturuluyor...",
    },
}

REC_TEXT_TO_KEY = {
    "Reduce clock frequency if system requirements allow.": "rec_clock",
    "Increase rise time slightly to reduce high-frequency harmonics.": "rec_rise",
    "Add a continuous ground plane.": "rec_ground",
    "Reduce the distance between signal traces and ground plane.": "rec_distance",
    "Shorten critical PCB traces.": "rec_trace",
    "Reduce stub length to minimize signal reflections.": "rec_stub",
    "Improve differential pair routing symmetry.": "rec_diff",
    "Apply shielding around critical circuits.": "rec_shield",
    "Improve filtering around the switching regulator.": "rec_switch",
}

FEATURE_TO_REC_KEY = {
    "Clock Frequency (MHz)": "rec_clock",
    "Rise Time (ns)": "rec_rise",
    "Ground Plane": "rec_ground",
    "Signal-GND Distance (mm)": "rec_distance",
    "Trace Length (mm)": "rec_trace",
    "Stub Length (mm)": "rec_stub",
    "Differential Pair Symmetry (%)": "rec_diff",
    "Shielding": "rec_shield",
    "Switching Regulator": "rec_switch",
    "Layer Count": "rec_ground",
}

EN_FEATURE_LABELS = [
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


def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("lang", "en")
    catalog = translations.get(lang, translations["en"])
    text = catalog.get(key) or translations["en"].get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text


def localize_band(band_en: str) -> str:
    return {
        "HIGH RISK": t("band_high"),
        "MEDIUM RISK": t("band_medium"),
        "LOW RISK": t("band_low"),
    }.get(band_en, band_en)


def localize_level(level: str) -> str:
    upper = str(level).upper()
    if "HIGH" in upper or "YÜKSEK" in upper or "YUKSEK" in upper:
        return t("level_high")
    if "MED" in upper or "ORTA" in upper:
        return t("level_medium")
    return t("level_low")


def localize_feature(en_name: str) -> str:
    return t(f"feat_{en_name}")


def localize_reason(en_name: str) -> str:
    return t(f"why_{en_name}")


def localize_rec_text(en_text: str) -> str:
    key = REC_TEXT_TO_KEY.get(en_text)
    return t(key) if key else en_text


def display_value(params: dict, en_feature: str) -> str:
    mapping = {
        "Layer Count": str(params["layer_count"]),
        "Clock Frequency (MHz)": f"{params['clock_frequency']} MHz",
        "Rise Time (ns)": f"{params['rise_time']} ns",
        "Signal-GND Distance (mm)": f"{params['signal_distance']} mm",
        "Trace Length (mm)": f"{params['trace_length']} mm",
        "Stub Length (mm)": f"{params['stub_length']} mm",
        "Differential Pair Symmetry (%)": f"{params['differential_pair']} %",
        "Ground Plane": (
            t("available") if params["ground_plane"] else t("not_available")
        ),
        "Shielding": t("applied") if params["shielding"] else t("not_applied"),
        "Switching Regulator": (
            t("present") if params["switching_regulator"] else t("not_present")
        ),
    }
    return mapping.get(en_feature, "")


def build_risk_cause_fix(
    prediction: str,
    risk_score: float,
    params: dict,
    shap_records: list[dict],
    recommendations: list[dict],
) -> list[dict]:
    is_fail = str(prediction).upper() == "FAIL"
    if is_fail:
        potential_risk = t("risk_fail")
    elif risk_score >= 40:
        potential_risk = t("risk_pass_elevated")
    else:
        potential_risk = t("risk_pass_low")

    rec_by_key = {
        REC_TEXT_TO_KEY.get(r["text"], ""): localize_rec_text(r["text"])
        for r in recommendations
    }

    cards: list[dict] = []
    ranked = sorted(
        shap_records, key=lambda r: float(r.get("Impact", 0)), reverse=True
    )
    for row in ranked:
        en_feat = row["Parameter"]
        impact = float(row["Impact"])
        level = str(row.get("Risk Level", ""))
        if impact < 0.03 and "HIGH" not in level.upper():
            continue

        rec_key = FEATURE_TO_REC_KEY.get(en_feat, "")
        fix = rec_by_key.get(rec_key) or localize_reason(en_feat)
        cause = (
            t("cause_prefix", feature=localize_feature(en_feat), impact=impact)
            + " "
            + localize_reason(en_feat)
            + " "
            + t("cause_value", value=display_value(params, en_feat))
        )
        cards.append(
            {
                "risk": potential_risk,
                "cause": cause,
                "fix": fix,
                "feature": en_feat,
                "impact": impact,
                "level": localize_level(level),
            }
        )
        if len(cards) >= 5:
            break

    if not cards and recommendations:
        for rec in recommendations[:5]:
            cards.append(
                {
                    "risk": potential_risk,
                    "cause": localize_rec_text(rec["text"]),
                    "fix": localize_rec_text(rec["text"]),
                    "feature": "",
                    "impact": 0.0,
                    "level": localize_level(rec["priority"]),
                }
            )
    return cards


def inject_industrial_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --emi-bg: #090A0F;
            --emi-panel: #12141A;
            --emi-panel-soft: #161821;
            --emi-border: #2A2D3A;
            --emi-text: #FFFFFF;
            --emi-muted: #8B949E;
            --emi-accent: #2563EB;
            --emi-accent-hover: #3B82F6;
            --emi-cyan: #00E5FF;
            --emi-success: #10B981;
            --emi-error: #EF4444;
            --emi-warn: #F59E0B;
            --emi-info: #00E5FF;
        }

        html, body, .stApp, .stMarkdown, .stText,
        button, input, select, textarea, label, p {
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
        }

        /* Never override Material icon fonts — expander chevrons break otherwise */
        .material-icons,
        .material-icons-outlined,
        .material-icons-round,
        .material-symbols-outlined,
        .material-symbols-rounded,
        .material-symbols-sharp,
        span[data-testid="stIconMaterial"],
        [data-testid="stExpander"] summary span[aria-hidden="true"],
        [data-testid="stExpanderToggleIcon"],
        [data-testid="stExpander"] svg {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                "Material Icons", "Material Icons Outlined" !important;
            font-style: normal !important;
            font-weight: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            speak: never;
            -webkit-font-smoothing: antialiased;
        }

        /* Clean expander headers: no stacked icon-name + Show/Hide text */
        [data-testid="stExpander"] details summary {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            gap: 0.5rem !important;
            list-style: none !important;
        }

        [data-testid="stExpander"] details summary p,
        [data-testid="stExpander"] details summary span:not([data-testid="stIconMaterial"]):not(.material-icons):not(.material-symbols-outlined):not(.material-symbols-rounded) {
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
        }

        [data-testid="stExpander"] details summary [data-testid="stExpanderToggleIcon"],
        [data-testid="stExpander"] details summary span[data-testid="stIconMaterial"] {
            flex: 0 0 auto;
            line-height: 1;
            overflow: hidden;
            color: #8B949E !important;
        }

        .stApp,
        .stApp > header,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"] {
            background: var(--emi-bg) !important;
            color: var(--emi-text);
        }

        /* Hide Streamlit chrome for a native desktop-app look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        footer,
        header {
            display: none !important;
            visibility: hidden !important;
        }

        .block-container {
            padding-top: 0 !important;
            padding-bottom: 1.5rem;
            max-width: 1500px;
        }

        h1, h2, h3, h4 {
            color: var(--emi-text) !important;
            white-space: normal;
            writing-mode: horizontal-tb;
        }

        p, label, span {
            color: var(--emi-text);
        }

        /* Seamless top-bar — flush with page, not a floating card */
        .emi-app-header {
            width: 100%;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            justify-content: center;
            background: #090A0F;
            padding: 1.1rem 0 1rem 0;
            border: none;
            border-bottom: 1px solid #00E5FF;
            border-radius: 0;
            margin: 0 0 1.35rem 0;
            box-shadow: none;
        }

        .emi-app-header h1 {
            margin: 0;
            padding: 0;
            color: #FFFFFF !important;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            line-height: 1.2;
            white-space: nowrap;
            writing-mode: horizontal-tb !important;
            text-orientation: mixed;
            width: 100%;
        }

        .emi-app-header p {
            margin: 0.3rem 0 0 0;
            padding: 0;
            color: #8B949E !important;
            font-size: 0.88rem;
            font-weight: 400;
            line-height: 1.4;
            white-space: normal;
            writing-mode: horizontal-tb !important;
            width: 100%;
        }

        div[data-testid="stMetric"] {
            background: var(--emi-panel);
            border: 1px solid var(--emi-border);
            border-radius: 4px;
            padding: 0.75rem 0.9rem;
            box-shadow: none;
        }

        div[data-testid="stMetric"] label {
            color: var(--emi-muted) !important;
        }

        .emi-input-panel,
        .emi-panel {
            background: var(--emi-panel);
            border: 1px solid var(--emi-border);
            border-radius: 4px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: none;
        }

        .emi-panel {
            border-left: 3px solid var(--emi-accent);
        }

        .emi-panel h4 {
            margin: 0 0 0.45rem 0;
            color: var(--emi-text);
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .emi-panel p {
            margin: 0.25rem 0;
            color: var(--emi-muted);
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .emi-label {
            color: var(--emi-cyan);
            font-weight: 700;
        }

        div[data-testid="stAlert"] {
            border-radius: 4px;
            border: 1px solid var(--emi-border);
        }

        /* Tabs — discrete clickable buttons, not run-on text */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.65rem !important;
            background: transparent !important;
            border: none !important;
            border-bottom: 1px solid #2A2D3A !important;
            border-radius: 0 !important;
            padding: 0 0 0.35rem 0 !important;
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab-list"] button,
        .stTabs button[data-baseweb="tab"],
        .stTabs [role="tab"] {
            background-color: #161821 !important;
            color: #8B949E !important;
            border: 1px solid #2A2D3A !important;
            border-bottom: none !important;
            border-radius: 6px 6px 0 0 !important;
            margin-right: 15px !important;
            padding: 10px 24px !important;
            min-height: 2.5rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.01em;
            white-space: nowrap !important;
            cursor: pointer !important;
            box-shadow: none !important;
            transition: background-color 0.15s ease, color 0.15s ease,
                border-color 0.15s ease;
        }

        .stTabs [data-baseweb="tab-list"] button:hover,
        .stTabs button[data-baseweb="tab"]:hover,
        .stTabs [role="tab"]:hover {
            background-color: #1E222D !important;
            color: #FFFFFF !important;
            border-color: #2563EB !important;
        }

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"],
        .stTabs button[data-baseweb="tab"][aria-selected="true"],
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #12141A !important;
            color: #FFFFFF !important;
            border-color: #2563EB !important;
            border-bottom: 2px solid #2563EB !important;
            box-shadow: inset 0 -2px 0 #2563EB !important;
        }

        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem !important;
        }

        /* Analyze / primary actions: Tech Blue — never emergency red */
        .stButton > button,
        .stFormSubmitButton > button,
        div[data-testid="stFormSubmitButton"] button,
        button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background: #2563EB !important;
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: 1px solid #1D4ED8 !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
            box-shadow: none !important;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover,
        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background: #3B82F6 !important;
            background-color: #3B82F6 !important;
            border-color: #2563EB !important;
            color: #FFFFFF !important;
        }

        .stButton > button:focus,
        .stFormSubmitButton > button:focus,
        div[data-testid="stFormSubmitButton"] button:focus {
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.35) !important;
        }

        /* Secondary / download buttons — metallic dark grey */
        button[kind="secondary"],
        button[data-testid="baseButton-secondary"] {
            background: #161821 !important;
            color: #FFFFFF !important;
            border: 1px solid #2A2D3A !important;
            border-radius: 4px !important;
        }

        button[kind="secondary"]:hover,
        button[data-testid="baseButton-secondary"]:hover {
            background: #1E222D !important;
            border-color: #2563EB !important;
            color: #FFFFFF !important;
        }

        /* Sliders → Tech Blue track / thumb */
        div[data-baseweb="slider"] div[role="slider"],
        [data-testid="stSlider"] [role="slider"] {
            background-color: #2563EB !important;
            border-color: #2563EB !important;
        }

        div[data-baseweb="slider"] div[data-testid="stTickBarMin"],
        div[data-baseweb="slider"] > div > div > div {
            background-image: none !important;
        }

        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background: #2A2D3A !important;
        }

        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {
            background: #2563EB !important;
        }

        /* Checkboxes → Tech Blue / emerald when checked */
        [data-testid="stCheckbox"] span[data-baseweb="checkbox"],
        [data-baseweb="checkbox"] > div {
            border-color: #2A2D3A !important;
        }

        [data-testid="stCheckbox"] [aria-checked="true"] span[data-baseweb="checkbox"],
        [data-baseweb="checkbox"][data-checked="true"] > div,
        label[data-baseweb="checkbox"] span[data-checked="true"] {
            background-color: #2563EB !important;
            border-color: #2563EB !important;
        }

        /* Radio (language) → Tech Blue selection */
        [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child,
        div[role="radiogroup"] label > div:first-child {
            border-color: #2A2D3A !important;
        }

        [data-testid="stRadio"] label[data-baseweb="radio"][data-checked="true"] > div:first-child,
        div[role="radiogroup"] label[aria-checked="true"] > div:first-child,
        [data-testid="stRadio"] [data-checked="true"] {
            border-color: #2563EB !important;
        }

        [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] div div,
        [data-baseweb="radio"][aria-checked="true"] > div:first-child > div {
            background-color: #2563EB !important;
        }

        /* Inputs / selects */
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] > div,
        .stTextInput input {
            background-color: #161821 !important;
            border-color: #2A2D3A !important;
            color: #FFFFFF !important;
            border-radius: 4px !important;
        }

        hr {
            border-color: var(--emi-border);
        }

        .stDataFrame, [data-testid="stDataFrame"] {
            border: 1px solid var(--emi-border);
            border-radius: 4px;
        }

        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        /* Control panel container — matte panel, sharp edge, no soft shadow */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #12141A !important;
            border: 1px solid #2A2D3A !important;
            border-radius: 4px !important;
            box-shadow: none !important;
        }

        .emi-panel-title {
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #FFFFFF;
            margin: 0 0 0.15rem 0;
        }

        .emi-panel-caption {
            font-size: 0.8rem;
            color: #8B949E;
            margin: 0 0 0.25rem 0;
        }

        .emi-group-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #00E5FF;
            margin: 0.15rem 0 0.1rem 0;
        }

        .emi-metrics {
            display: flex;
            gap: 0.75rem;
            margin: 0.25rem 0 1rem 0;
        }

        .emi-metric-card {
            background: #12141A;
            border: 1px solid #2A2D3A;
            border-top: 2px solid #2563EB;
            border-radius: 4px;
            padding: 0.85rem 0.95rem;
            min-height: 104px;
            flex: 1 1 0;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
            box-shadow: none;
        }

        .emi-metric-label {
            color: #8B949E;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .emi-metric-value {
            color: #FFFFFF;
            font-size: 1.7rem;
            font-weight: 700;
            line-height: 1.1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .emi-metric-sub {
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Red reserved for FAIL / high-risk only */
        .emi-metric-card.tone-fail { border-top-color: #EF4444; }
        .emi-metric-card.tone-pass { border-top-color: #10B981; }
        .emi-metric-card.tone-warn { border-top-color: #F59E0B; }
        .emi-metric-card.tone-info { border-top-color: #2563EB; }
        .emi-sub-fail { color: #EF4444; }
        .emi-sub-pass { color: #10B981; }
        .emi-sub-warn { color: #F59E0B; }
        .emi-sub-muted { color: #8B949E; }

        /* Progress bar → tech blue fill */
        .stProgress > div > div > div > div {
            background-color: #2563EB !important;
        }

        @media (max-width: 980px) {
            .emi-metrics {
                flex-wrap: wrap;
            }
            .emi-metric-card {
                flex-basis: calc(50% - 0.75rem);
            }
            .emi-app-header h1 {
                white-space: normal;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    """Seamless deep-tech top-bar (bilingual). Same bg as page; cyan rule only."""
    import html as _html

    title = _html.escape(t("app_title"))
    subtitle = _html.escape(t("header_subtitle"))
    st.markdown(
        f"""
        <div class="emi-app-header">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rcf_card(card: dict) -> None:
    title = (
        localize_feature(card["feature"]) if card["feature"] else card["level"]
    )
    st.markdown(
        f"""
        <div class="emi-panel">
          <h4>{card["level"]} · {title}</h4>
          <p><span class="emi-label">{t("rcf_risk")}:</span> {card["risk"]}</p>
          <p><span class="emi-label">{t("rcf_cause")}:</span> {card["cause"]}</p>
          <p><span class="emi-label">{t("rcf_fix")}:</span> {card["fix"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(cards: list[dict]) -> None:
    """Render equal-size metric cards via a single CSS grid block."""
    import html as _html

    blocks = ['<div class="emi-metrics">']
    for c in cards:
        tone = c.get("tone", "info")
        sub_class = c.get("sub_class", "emi-sub-muted")
        label = _html.escape(str(c["label"]))
        value = _html.escape(str(c["value"]))
        sub = _html.escape(str(c.get("sub", "")))
        blocks.append(
            f'<div class="emi-metric-card tone-{tone}">'
            f'<div class="emi-metric-label" title="{label}">{label}</div>'
            f'<div class="emi-metric-value" title="{value}">{value}</div>'
            f'<div class="emi-metric-sub {sub_class}" title="{sub}">{sub}</div>'
            f"</div>"
        )
    blocks.append("</div>")
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_pass_fail_alert(
    prediction: str, confidence: float, risk_score: float
) -> None:
    band_en, _ = risk_band(risk_score)
    band = localize_band(band_en)
    is_fail = str(prediction).upper() == "FAIL"

    if is_fail:
        st.error(
            f"**{t('fail_title')}**\n\n"
            f"{t('fail_body', band=band, score=risk_score)}"
        )
        if confidence >= 80:
            st.caption(t("certainty_high", c=confidence))
        elif confidence < 60:
            st.caption(t("certainty_mod_fail", c=confidence))
    else:
        st.success(
            f"**{t('pass_title')}**\n\n"
            f"{t('pass_body', band=band, score=risk_score)}"
        )
        if risk_score >= 40:
            st.warning(t("pass_residual"))
        if confidence < 60:
            st.caption(t("certainty_mod_pass", c=confidence))


st.set_page_config(
    page_title="EMI Risk Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_industrial_css()


# ---------------------------------------------------------------------------
# Caching layer — heavy resources stay in memory / data cache
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Load the RandomForest .pkl once; reuse from RAM on every rerun."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file missing: {MODEL_PATH}. Run src/train_model.py first."
        )
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_cached_explainer():
    """Build TreeExplainer once against the cached model."""
    model = get_cached_model()
    return build_explainer(model)


@st.cache_data(show_spinner=False)
def load_reference_dataset() -> pd.DataFrame:
    """Cache static reference CSV (training / baseline statistics)."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset missing: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_data(show_spinner=False)
def cached_analyze(
    layer_count: int,
    clock_frequency: float,
    rise_time: float,
    signal_distance: float,
    trace_length: float,
    stub_length: float,
    differential_pair: float,
    ground_plane: bool,
    shielding: bool,
    switching_regulator: bool,
) -> dict:
    """
    Cache prediction + SHAP for identical PCB inputs.
    Model/explainer come from cache_resource (not reloaded from disk).
    """
    model = get_cached_model()
    explainer = get_cached_explainer()
    params = {
        "layer_count": layer_count,
        "clock_frequency": clock_frequency,
        "rise_time": rise_time,
        "signal_distance": signal_distance,
        "trace_length": trace_length,
        "stub_length": stub_length,
        "differential_pair": differential_pair,
        "ground_plane": ground_plane,
        "shielding": shielding,
        "switching_regulator": switching_regulator,
    }
    result = analyze_pcb(model, explainer, params)
    return {
        "warnings": result["warnings"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "risk_score": result["risk_score"],
        "band_label": result["band_label"],
        "shap_records": result["shap_df"].to_dict(orient="records"),
        "recommendations": result["recommendations"],
        "high_count": result["high_count"],
        "params": params,
    }


@st.cache_data(show_spinner=False)
def cached_pdf(
    prediction: str,
    confidence: float,
    risk_score: float,
    shap_records: tuple,
    recommendations: tuple,
    params_items: tuple,
) -> bytes:
    """Build PDF bytes once per analysis fingerprint."""
    params = dict(params_items)
    buffer = generate_emi_pdf_report(
        pcb_inputs=params,
        prediction=prediction,
        confidence=confidence,
        risk_score=risk_score,
        shap_rows=[dict(row) for row in shap_records],
        recommendations=[dict(rec) for rec in recommendations],
    )
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def cached_shap_chart_png(
    labels: tuple,
    impacts: tuple,
    title: str,
    xlabel: str,
) -> bytes:
    """Cache matplotlib SHAP chart as PNG bytes (avoids redraw on every widget rerun)."""
    import io

    fig, ax = plt.subplots(figsize=(8, 4.0))
    colors = [impact_color(float(v)) for v in impacts]
    ax.barh(list(labels), list(impacts), color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="#f8fafc")
    ax.xaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Bootstrap — warm caches once (no spinner noise on subsequent reruns)
# ---------------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

try:
    get_cached_model()
    get_cached_explainer()
    if DATASET_PATH.exists():
        load_reference_dataset()
except Exception as exc:  # noqa: BLE001
    st.error(f"{t('model_load_failed')}: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Integrated layout: premium header + compact input grid + dashboard
# ---------------------------------------------------------------------------
render_app_header()

with st.container(border=True):
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown(
            f"<div class='emi-panel-title'>{t('control_panel')}</div>"
            f"<div class='emi-panel-caption'>{t('sidebar_caption')}</div>",
            unsafe_allow_html=True,
        )
    with head_r:
        # Language stays OUTSIDE the form so switching is instant.
        st.radio(
            t("language"),
            options=["en", "tr"],
            format_func=lambda code: "English" if code == "en" else "Türkçe",
            horizontal=True,
            key="lang",
            label_visibility="collapsed",
        )

    # Inputs live inside a form: typing/toggling does NOT rerun the app.
    # A single rerun happens only when the submit button is pressed.
    with st.form("pcb_form", border=False):
        row1 = st.columns(4)
        with row1[0]:
            layer_count = st.selectbox(
                t("layer_count"),
                list(PCB_BOUNDS["layer_count"]["allowed"]),
                index=2,
            )
        with row1[1]:
            clock_frequency = st.number_input(
                t("clock_frequency"),
                min_value=float(PCB_BOUNDS["clock_frequency"]["min"]),
                max_value=float(PCB_BOUNDS["clock_frequency"]["max"]),
                value=300.0,
                help=t("help_clock"),
            )
        with row1[2]:
            rise_time = st.number_input(
                t("rise_time"),
                min_value=float(PCB_BOUNDS["rise_time"]["min"]),
                max_value=float(PCB_BOUNDS["rise_time"]["max"]),
                value=2.0,
                help=t("help_rise"),
            )
        with row1[3]:
            signal_distance = st.number_input(
                t("signal_distance"),
                min_value=float(PCB_BOUNDS["signal_distance"]["min"]),
                max_value=float(PCB_BOUNDS["signal_distance"]["max"]),
                value=1.0,
            )

        row2 = st.columns(4)
        with row2[0]:
            trace_length = st.number_input(
                t("trace_length"),
                min_value=float(PCB_BOUNDS["trace_length"]["min"]),
                max_value=float(PCB_BOUNDS["trace_length"]["max"]),
                value=100.0,
            )
        with row2[1]:
            stub_length = st.number_input(
                t("stub_length"),
                min_value=float(PCB_BOUNDS["stub_length"]["min"]),
                max_value=float(PCB_BOUNDS["stub_length"]["max"]),
                value=2.0,
            )
        with row2[2]:
            differential_pair = st.slider(
                t("differential_pair"),
                float(PCB_BOUNDS["differential_pair"]["min"]),
                float(PCB_BOUNDS["differential_pair"]["max"]),
                95.0,
            )
        with row2[3]:
            st.markdown(
                f"<div class='emi-group-label'>{t('section_mitigation')}</div>",
                unsafe_allow_html=True,
            )
            ground_plane = st.checkbox(t("ground_plane"), value=True)
            shielding = st.checkbox(t("shielding"), value=True)
            switching_regulator = st.checkbox(
                t("switching_regulator"), value=False
            )

        analyze = st.form_submit_button(
            t("analyze"), use_container_width=True, type="primary"
        )

col_dashboard = st.container()

current_inputs = {
    "layer_count": int(layer_count),
    "clock_frequency": float(clock_frequency),
    "rise_time": float(rise_time),
    "signal_distance": float(signal_distance),
    "trace_length": float(trace_length),
    "stub_length": float(stub_length),
    "differential_pair": float(differential_pair),
    "ground_plane": bool(ground_plane),
    "shielding": bool(shielding),
    "switching_regulator": bool(switching_regulator),
}

if analyze:
    try:
        validate_pcb_inputs(current_inputs)
        with st.spinner(t("running_analysis")):
            st.session_state["analysis_result"] = cached_analyze(
                current_inputs["layer_count"],
                current_inputs["clock_frequency"],
                current_inputs["rise_time"],
                current_inputs["signal_distance"],
                current_inputs["trace_length"],
                current_inputs["stub_length"],
                current_inputs["differential_pair"],
                current_inputs["ground_plane"],
                current_inputs["shielding"],
                current_inputs["switching_regulator"],
            )
        st.session_state["analyzed"] = True
        st.session_state["pcb_inputs"] = current_inputs
    except PCBValidationError as exc:
        with col_dashboard:
            st.error(f"{t('invalid_params')}: {exc}")
        st.stop()

with col_dashboard:
    if not st.session_state.get("analyzed"):
        st.info(t("idle_info"))
        st.stop()

    p = st.session_state["pcb_inputs"]

    if p != current_inputs:
        st.warning(t("stale_warning"))

    # Prefer session snapshot; fall back to data-cache (RAM, no disk reload)
    result = st.session_state.get("analysis_result")
    if result is None or result.get("params") != p:
        try:
            result = cached_analyze(
                p["layer_count"],
                p["clock_frequency"],
                p["rise_time"],
                p["signal_distance"],
                p["trace_length"],
                p["stub_length"],
                p["differential_pair"],
                p["ground_plane"],
                p["shielding"],
                p["switching_regulator"],
            )
            st.session_state["analysis_result"] = result
        except PCBValidationError as exc:
            st.error(f"{t('validation_failed')}: {exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"{t('analysis_failed')}: {exc}")
            st.stop()

    for warning in result.get("warnings", []):
        if "exceeds the typical training range" in warning:
            st.warning(
                t("ood_clock", warn_max=PCB_BOUNDS["clock_frequency"]["warn_max"])
            )
        else:
            st.warning(warning)

    prediction = result["prediction"]
    confidence = result["confidence"]
    risk_score = result["risk_score"]
    band_label = localize_band(result["band_label"])
    high_count = result["high_count"]
    recommendations = result["recommendations"]
    shap_records = result["shap_records"]

    input_summary = pd.DataFrame(
        {
            t("col_parameter"): [localize_feature(f) for f in EN_FEATURE_LABELS],
            t("col_value"): [display_value(p, f) for f in EN_FEATURE_LABELS],
        }
    )

    shap_display = pd.DataFrame(
        {
            t("col_parameter"): [
                localize_feature(r["Parameter"]) for r in shap_records
            ],
            t("col_impact"): [r["Impact"] for r in shap_records],
            t("col_level"): [localize_level(r["Risk Level"]) for r in shap_records],
            t("col_reason"): [
                localize_reason(r["Parameter"]) for r in shap_records
            ],
        }
    )

    try:
        pdf_bytes = cached_pdf(
            prediction=str(prediction),
            confidence=float(confidence),
            risk_score=float(risk_score),
            shap_records=tuple(
                tuple(sorted(row.items())) for row in shap_records
            ),
            recommendations=tuple(
                tuple(sorted(rec.items())) for rec in recommendations
            ),
            params_items=tuple(sorted(p.items())),
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"{t('pdf_failed')}: {exc}")
        pdf_bytes = None

    render_pass_fail_alert(str(prediction), confidence, risk_score)

    is_fail = str(prediction).upper() == "FAIL"
    render_metric_cards(
        [
            {
                "label": t("metric_prediction"),
                "value": str(prediction).upper(),
                "sub": t("delta_fail") if is_fail else t("delta_pass"),
                "tone": "fail" if is_fail else "pass",
                "sub_class": "emi-sub-fail" if is_fail else "emi-sub-pass",
            },
            {
                "label": t("metric_risk"),
                "value": f"{risk_score:.1f}",
                "sub": band_label,
                "tone": "warn" if risk_score >= 40 else "pass",
                "sub_class": (
                    "emi-sub-warn" if risk_score >= 40 else "emi-sub-pass"
                ),
            },
            {
                "label": t("metric_confidence"),
                "value": f"{confidence:.1f}%",
                "sub": "",
                "tone": "info",
                "sub_class": "emi-sub-muted",
            },
            {
                "label": t("metric_fixes"),
                "value": str(high_count),
                "sub": t("delta_action") if high_count else t("delta_none"),
                "tone": "fail" if high_count else "info",
                "sub_class": (
                    "emi-sub-fail" if high_count else "emi-sub-muted"
                ),
            },
        ]
    )

    st.progress(min(max(risk_score / 100.0, 0.0), 1.0))
    st.caption(t("risk_progress", score=risk_score))

    if pdf_bytes is not None:
        st.download_button(
            label=t("download_pdf"),
            data=pdf_bytes,
            file_name="EMI_Risk_Analysis_Report.pdf",
            mime="application/pdf",
            use_container_width=False,
        )

    st.divider()

    st.subheader(t("rcf_title"))
    st.caption(t("rcf_intro"))
    rcf_cards = build_risk_cause_fix(
        prediction, risk_score, p, shap_records, recommendations
    )
    if not rcf_cards:
        st.success(t("rcf_empty"))
    else:
        for idx, card in enumerate(rcf_cards, start=1):
            title = (
                localize_feature(card["feature"])
                if card["feature"]
                else card["level"]
            )
            with st.expander(
                f"{idx}. [{card['level']}] {title}", expanded=(idx == 1)
            ):
                render_rcf_card(card)

    st.divider()

    tab_pcb, tab_shap, tab_recs = st.tabs(
        [t("tab_pcb"), t("tab_shap"), t("tab_recs")]
    )

    with tab_pcb:
        st.subheader(t("pcb_inputs"))
        st.dataframe(input_summary, use_container_width=True, hide_index=True)
        with st.expander(t("eng_notes"), expanded=False):
            for en_name in EN_FEATURE_LABELS:
                st.markdown(
                    f"**{localize_feature(en_name)}** — {localize_reason(en_name)}"
                )

    with tab_shap:
        st.subheader(t("shap_ranking"))
        st.dataframe(shap_display, use_container_width=True, hide_index=True)
        with st.expander(t("shap_chart"), expanded=True):
            chart_labels = tuple(
                localize_feature(r["Parameter"]) for r in shap_records
            )
            chart_impacts = tuple(float(r["Impact"]) for r in shap_records)
            chart_png = cached_shap_chart_png(
                chart_labels,
                chart_impacts,
                t("shap_title"),
                t("shap_xlabel"),
            )
            st.image(chart_png, use_container_width=True)

    with tab_recs:
        st.subheader(t("mitigation"))
        if not recommendations:
            st.success(t("no_recs"))
        else:
            high = [r for r in recommendations if r["priority"] == "High"]
            medium = [r for r in recommendations if r["priority"] == "Medium"]
            low = [r for r in recommendations if r["priority"] == "Low"]

            with st.expander(
                f"{t('priority_high')} ({len(high)})", expanded=bool(high)
            ):
                if not high:
                    st.caption(t("no_high"))
                for rec in high:
                    st.error(
                        f"**{t('priority_high')}:** {localize_rec_text(rec['text'])}"
                    )

            with st.expander(
                f"{t('priority_medium')} ({len(medium)})",
                expanded=bool(medium) and not high,
            ):
                if not medium:
                    st.caption(t("no_medium"))
                for rec in medium:
                    st.warning(
                        f"**{t('priority_medium')}:** {localize_rec_text(rec['text'])}"
                    )

            with st.expander(
                f"{t('priority_low')} ({len(low)})", expanded=False
            ):
                if not low:
                    st.caption(t("no_low"))
                for rec in low:
                    st.info(
                        f"**{t('priority_low')}:** {localize_rec_text(rec['text'])}"
                    )
