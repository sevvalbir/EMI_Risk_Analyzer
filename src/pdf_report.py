"""EMI Risk Analyzer — professional PDF report generation (in-memory)."""

from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from fpdf import FPDF


def _dejavu_font_paths() -> tuple[Path, Path]:
    """Resolve DejaVu fonts shipped with matplotlib (Unicode-safe, Cloud-friendly)."""
    import matplotlib

    font_dir = Path(matplotlib.__file__).resolve().parent / "mpl-data" / "fonts" / "ttf"
    regular = font_dir / "DejaVuSans.ttf"
    bold = font_dir / "DejaVuSans-Bold.ttf"
    if not regular.exists():
        raise FileNotFoundError(
            f"DejaVuSans.ttf not found at {regular}. "
            "matplotlib must be installed for PDF Unicode fonts."
        )
    if not bold.exists():
        bold = regular
    return regular, bold


def _clean(text: Any) -> str:
    """Strip emojis / non-printable symbols for stable PDF rendering."""
    s = str(text)
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
    s = re.sub(
        r"[\u2600-\u27BF\uFE0F\u200D\u2300-\u23FF\u2B50\u25A0-\u25FF]",
        "",
        s,
    )
    return " ".join(s.split()).strip()


def _risk_band(risk_score: float) -> str:
    if risk_score >= 70:
        return "HIGH RISK"
    if risk_score >= 40:
        return "MEDIUM RISK"
    return "LOW RISK"


class EMIPDFReport(FPDF):
    """Branded multi-page EMI analysis report."""

    NAVY = (15, 40, 70)
    ACCENT = (0, 120, 160)
    LIGHT = (245, 248, 250)
    FAIL = (192, 57, 43)
    PASS = (39, 174, 96)
    WARN = (211, 84, 0)
    MUTED = (90, 100, 110)

    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        regular, bold = _dejavu_font_paths()
        self.add_font("DejaVu", "", str(regular))
        self.add_font("DejaVu", "B", str(bold))
        self._section = "EMI Risk Analyzer Report"

    def header(self) -> None:
        self.set_fill_color(*self.NAVY)
        self.rect(0, 0, 210, 18, "F")
        self.set_xy(12, 5)
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "EMI Risk Analyzer  |  PCB Decision Support Report", align="L")
        self.set_y(22)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_draw_color(*self.ACCENT)
        self.set_line_width(0.3)
        self.line(12, self.get_y(), 198, self.get_y())
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*self.MUTED)
        self.cell(
            0,
            8,
            f"Confidential engineering report  ·  Page {self.page_no()}/{{nb}}",
            align="C",
        )

    def section_title(self, title: str) -> None:
        self.ln(2)
        self.set_fill_color(*self.ACCENT)
        self.rect(12, self.get_y(), 2.5, 7, "F")
        self.set_x(17)
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(*self.NAVY)
        self.cell(0, 7, _clean(title), ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def key_value_table(
        self,
        rows: Sequence[tuple[str, str]],
        col1: float = 85,
        col2: float = 95,
    ) -> None:
        self.set_font("DejaVu", "B", 9)
        self.set_fill_color(*self.NAVY)
        self.set_text_color(255, 255, 255)
        self.set_x(12)
        self.cell(col1, 7, "Parameter", border=0, fill=True)
        self.cell(col2, 7, "Value", border=0, fill=True, ln=True)
        self.set_text_color(0, 0, 0)

        for i, (k, v) in enumerate(rows):
            fill = i % 2 == 0
            if fill:
                self.set_fill_color(*self.LIGHT)
            self.set_x(12)
            self.set_font("DejaVu", "", 9)
            self.cell(col1, 6.5, _clean(k), border=0, fill=fill)
            self.set_font("DejaVu", "B", 9)
            self.cell(col2, 6.5, _clean(v), border=0, fill=fill, ln=True)

    def status_banner(
        self,
        prediction: str,
        confidence: float,
        risk_score: float,
    ) -> None:
        is_fail = str(prediction).upper() == "FAIL"
        color = self.FAIL if is_fail else self.PASS
        self.set_fill_color(*color)
        self.set_x(12)
        self.cell(186, 14, "", fill=True, ln=True)
        y = self.get_y() - 14
        self.set_xy(16, y + 2)
        self.set_font("DejaVu", "B", 16)
        self.set_text_color(255, 255, 255)
        self.cell(70, 10, f"Prediction: {str(prediction).upper()}")
        self.set_font("DejaVu", "", 10)
        self.cell(55, 10, f"Confidence: {confidence:.2f}%")
        self.cell(55, 10, f"EMI Risk: {risk_score:.1f}/100")
        self.set_text_color(0, 0, 0)
        self.set_y(y + 16)

        band = _risk_band(risk_score)
        if risk_score >= 70:
            band_color = self.FAIL
        elif risk_score >= 40:
            band_color = self.WARN
        else:
            band_color = self.PASS
        self.set_fill_color(*band_color)
        self.set_x(12)
        self.set_font("DejaVu", "B", 10)
        self.set_text_color(255, 255, 255)
        self.cell(186, 8, f"Risk Classification: {band}", fill=True, align="C", ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)


def generate_emi_pdf_report(
    pcb_inputs: Mapping[str, Any],
    prediction: str,
    confidence: float,
    risk_score: float,
    shap_rows: Sequence[Mapping[str, Any]],
    recommendations: Sequence[Mapping[str, str]],
) -> io.BytesIO:
    """
    Build a professional EMI analysis PDF entirely in memory.

    Parameters
    ----------
    pcb_inputs:
        Dict with layer_count, clock_frequency, rise_time, signal_distance,
        trace_length, stub_length, differential_pair, ground_plane, shielding,
        switching_regulator.
    prediction:
        "PASS" or "FAIL".
    confidence:
        Model confidence percentage (0-100).
    risk_score:
        EMI risk score / FAIL probability percentage (0-100).
    shap_rows:
        Sequence of dicts with keys: Parameter, Impact, Risk Level,
        Fiziksel Gerekçe (Why?)  [or "Reason"].
    recommendations:
        Sequence of dicts with keys: priority ("High"|"Medium"|"Low"), text.

    Returns
    -------
    io.BytesIO
        PDF bytes ready for st.download_button.
    """
    pdf = EMIPDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Cover / meta ---
    pdf.set_font("DejaVu", "B", 18)
    pdf.set_text_color(*pdf.NAVY)
    pdf.set_x(12)
    pdf.cell(0, 10, "EMI Risk Analysis Report", ln=True)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(*pdf.MUTED)
    pdf.set_x(12)
    pdf.cell(
        0,
        5,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
        "AI-assisted PCB EMI decision support",
        ln=True,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.status_banner(prediction, confidence, risk_score)

    # --- PCB parameters ---
    pdf.section_title("1. PCB Design Parameters")
    param_rows = [
        ("Layer Count", str(pcb_inputs.get("layer_count", ""))),
        ("Clock Frequency (MHz)", str(pcb_inputs.get("clock_frequency", ""))),
        ("Rise Time (ns)", str(pcb_inputs.get("rise_time", ""))),
        ("Signal-GND Distance (mm)", str(pcb_inputs.get("signal_distance", ""))),
        ("Trace Length (mm)", str(pcb_inputs.get("trace_length", ""))),
        ("Stub Length (mm)", str(pcb_inputs.get("stub_length", ""))),
        (
            "Differential Pair Symmetry (%)",
            str(pcb_inputs.get("differential_pair", "")),
        ),
        (
            "Ground Plane",
            "Available" if pcb_inputs.get("ground_plane") else "Not Available",
        ),
        (
            "Shielding",
            "Applied" if pcb_inputs.get("shielding") else "Not Applied",
        ),
        (
            "Switching Regulator",
            "Present" if pcb_inputs.get("switching_regulator") else "Not Present",
        ),
    ]
    pdf.key_value_table(param_rows)

    # --- Prediction summary ---
    pdf.section_title("2. Model Prediction Summary")
    pdf.set_font("DejaVu", "", 9)
    pdf.set_x(12)
    summary = (
        f"The Random Forest classifier predicts {str(prediction).upper()} with "
        f"{confidence:.2f}% confidence. The EMI Risk Score "
        f"({risk_score:.1f}/100) is the model's estimated FAIL probability. "
        f"Classification band: {_risk_band(risk_score)}."
    )
    pdf.multi_cell(186, 5, _clean(summary))
    pdf.ln(2)

    # --- SHAP factors ---
    pdf.section_title("3. SHAP-Based Risk Factors")
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*pdf.MUTED)
    pdf.set_x(12)
    pdf.multi_cell(
        186,
        4.5,
        "Absolute SHAP impact ranks how strongly each PCB parameter influenced "
        "the EMI prediction for this design. Higher impact = stronger driver.",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    # Table header
    w_param, w_impact, w_level = 55, 22, 28
    w_reason = 186 - (w_param + w_impact + w_level)
    pdf.set_font("DejaVu", "B", 8)
    pdf.set_fill_color(*pdf.NAVY)
    pdf.set_text_color(255, 255, 255)
    pdf.set_x(12)
    pdf.cell(w_param, 6.5, "Parameter", fill=True)
    pdf.cell(w_impact, 6.5, "Impact", fill=True, align="C")
    pdf.cell(w_level, 6.5, "Level", fill=True, align="C")
    pdf.cell(w_reason, 6.5, "Engineering Rationale", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)

    for i, row in enumerate(shap_rows):
        param = _clean(row.get("Parameter", ""))
        impact = row.get("Impact", 0)
        try:
            impact_str = f"{float(impact):.4f}"
        except (TypeError, ValueError):
            impact_str = _clean(impact)
        level = _clean(
            str(row.get("Risk Level", ""))
            .replace("HIGH", "HIGH")
            .replace("MEDIUM", "MED")
            .replace("LOW", "LOW")
        )
        # Keep only HIGH/MEDIUM/LOW words
        level_short = "LOW"
        level_upper = level.upper()
        if "HIGH" in level_upper:
            level_short = "HIGH"
        elif "MED" in level_upper:
            level_short = "MEDIUM"
        reason = _clean(
            row.get("Fiziksel Gerekçe (Why?)")
            or row.get("Reason")
            or row.get("Engineering Rationale")
            or ""
        )

        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(*pdf.LIGHT)

        # Estimate row height from wrapped reason
        pdf.set_font("DejaVu", "", 7.5)
        # Use multi_cell in a controlled way: print param/impact/level on first line,
        # reason may wrap.
        x0, y0 = 12, pdf.get_y()
        if y0 > 260:
            pdf.add_page()
            y0 = pdf.get_y()

        line_h = 4.2
        # Reason height
        reason_lines = pdf.multi_cell(w_reason, line_h, reason, dry_run=True, output="LINES")
        n_lines = max(1, len(reason_lines))
        row_h = max(6.5, n_lines * line_h + 1)

        if fill:
            pdf.set_fill_color(*pdf.LIGHT)
            pdf.rect(12, y0, 186, row_h, "F")

        pdf.set_xy(x0, y0 + 0.8)
        pdf.set_font("DejaVu", "B", 7.5)
        pdf.cell(w_param, line_h, param[:42])
        pdf.set_font("DejaVu", "", 7.5)
        pdf.cell(w_impact, line_h, impact_str, align="C")
        pdf.set_font("DejaVu", "B", 7.5)
        if level_short == "HIGH":
            pdf.set_text_color(*pdf.FAIL)
        elif level_short == "MEDIUM":
            pdf.set_text_color(*pdf.WARN)
        else:
            pdf.set_text_color(*pdf.PASS)
        pdf.cell(w_level, line_h, level_short, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.set_xy(x0 + w_param + w_impact + w_level, y0 + 0.8)
        pdf.set_font("DejaVu", "", 7)
        pdf.multi_cell(w_reason, line_h, reason)
        pdf.set_y(y0 + row_h)

    # --- Recommendations ---
    if pdf.get_y() > 230:
        pdf.add_page()

    pdf.section_title("4. Engineering Design Recommendations")
    if not recommendations:
        pdf.set_font("DejaVu", "", 9)
        pdf.set_x(12)
        pdf.multi_cell(
            186,
            5,
            "No major EMI improvements are required for the current parameter set.",
        )
    else:
        priority_colors = {
            "High": pdf.FAIL,
            "Medium": pdf.WARN,
            "Low": pdf.PASS,
        }
        for i, rec in enumerate(recommendations, start=1):
            priority = str(rec.get("priority", "Low")).capitalize()
            if priority not in priority_colors:
                priority = "Low"
            text = _clean(rec.get("text", ""))
            # Strip leading priority labels if already present
            text = re.sub(
                r"^(High|Medium|Low)\s*Priority\s*:\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            if pdf.get_y() > 270:
                pdf.add_page()

            color = priority_colors[priority]
            pdf.set_fill_color(*color)
            pdf.set_x(12)
            pdf.cell(3, 7, "", fill=True)
            pdf.set_font("DejaVu", "B", 9)
            pdf.set_text_color(*color)
            pdf.cell(30, 7, f"{priority} Priority")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("DejaVu", "", 9)
            pdf.set_x(45)
            # multi_cell for long advice
            y_before = pdf.get_y()
            pdf.multi_cell(153, 5, f"{i}. {text}")
            # Ensure spacing after wrapped text
            if pdf.get_y() < y_before + 7:
                pdf.set_y(y_before + 7)
            pdf.ln(1)

    # --- Disclaimer ---
    pdf.ln(4)
    pdf.set_draw_color(*pdf.MUTED)
    pdf.set_line_width(0.2)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("DejaVu", "", 7.5)
    pdf.set_text_color(*pdf.MUTED)
    pdf.set_x(12)
    pdf.multi_cell(
        186,
        4,
        "Disclaimer: This report is a decision-support output from a machine-learning "
        "model trained on synthetic / historical PCB EMI features. It does not replace "
        "lab EMC testing, normative compliance (e.g. CISPR / FCC), or professional "
        "hardware review. Validate critical designs with measurement and expert sign-off.",
    )

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer
