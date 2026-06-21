"""
Report generation: PDF, CSV, Excel.
"""

import io
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from config import THEME, EMOTION_LABELS


# ─── CSV Report ───────────────────────────────────────────────────────────────

def generate_csv_report(analysis_result: Dict) -> bytes:
    rows = [
        {"Field": "Predicted Emotion", "Value": analysis_result.get("emotion", "N/A")},
        {"Field": "Confidence", "Value": f"{analysis_result.get('confidence', 0):.2%}"},
        {"Field": "Model", "Value": analysis_result.get("model_name", "SER Model")},
        {"Field": "Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Field": "Duration (s)", "Value": analysis_result.get("duration", "N/A")},
        {"Field": "Sample Rate", "Value": analysis_result.get("sample_rate", "N/A")},
    ]

    probs = analysis_result.get("probabilities", {})
    for emotion, prob in probs.items():
        rows.append({"Field": f"Prob: {emotion.capitalize()}", "Value": f"{prob:.4f}"})

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()


# ─── Excel Report ─────────────────────────────────────────────────────────────

def generate_excel_report(
    analysis_result: Dict,
    eval_metrics: Optional[Dict] = None,
) -> bytes:
    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Prediction
        pred_data = {
            "Metric": ["Predicted Emotion", "Confidence", "Model", "Timestamp"],
            "Value": [
                analysis_result.get("emotion", "N/A"),
                f"{analysis_result.get('confidence', 0):.2%}",
                analysis_result.get("model_name", "SER Model"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ],
        }
        pd.DataFrame(pred_data).to_excel(writer, sheet_name="Prediction", index=False)

        # Sheet 2: Probabilities
        probs = analysis_result.get("probabilities", {})
        if probs:
            prob_data = {
                "Emotion": [e.capitalize() for e in probs],
                "Probability": [f"{p:.4f}" for p in probs.values()],
                "Percentage": [f"{p*100:.1f}%" for p in probs.values()],
            }
            pd.DataFrame(prob_data).to_excel(writer, sheet_name="Probabilities", index=False)

        # Sheet 3: Model Metrics
        if eval_metrics:
            metrics_data = {
                "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"],
                "Score": [
                    f"{eval_metrics.get('accuracy', 0):.4f}",
                    f"{eval_metrics.get('precision_macro', 0):.4f}",
                    f"{eval_metrics.get('recall_macro', 0):.4f}",
                    f"{eval_metrics.get('f1_macro', 0):.4f}",
                ],
            }
            pd.DataFrame(metrics_data).to_excel(writer, sheet_name="Model Metrics", index=False)

    return buf.getvalue()


# ─── PDF Report ───────────────────────────────────────────────────────────────

def generate_pdf_report(
    analysis_result: Dict,
    eval_metrics: Optional[Dict] = None,
    audio_info: Optional[Dict] = None,
) -> bytes:
    if not HAS_REPORTLAB:
        # Fallback to plain-text PDF via fpdf2 or simple text
        return _generate_text_pdf(analysis_result)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    green = rl_colors.HexColor(THEME["primary"])
    light_green = rl_colors.HexColor(THEME["accent"])

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        textColor=green, fontSize=20, spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        textColor=green, fontSize=13, spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, spaceAfter=4,
    )

    story = []

    # Header
    story.append(Paragraph("AI-Powered Speech Emotion Recognition", title_style))
    story.append(Paragraph("Analysis Report", h2_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=green, spaceAfter=12))

    # Prediction section
    story.append(Paragraph("Prediction Summary", h2_style))
    emotion = analysis_result.get("emotion", "N/A").capitalize()
    confidence = analysis_result.get("confidence", 0)

    pred_data = [
        ["Field", "Value"],
        ["Detected Emotion", emotion],
        ["Confidence Score", f"{confidence:.2%}"],
        ["Model Used", analysis_result.get("model_name", "SER CNN")],
        ["Analysis Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    if audio_info:
        pred_data += [
            ["Audio Duration", f"{audio_info.get('duration_sec', 0):.2f}s"],
            ["Sample Rate", f"{audio_info.get('sample_rate', 22050)} Hz"],
        ]

    t = Table(pred_data, colWidths=[7*cm, 9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), green),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F0EDE6")]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#D4CFC7")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Emotion probabilities
    probs = analysis_result.get("probabilities", {})
    if probs:
        story.append(Paragraph("Emotion Probability Scores", h2_style))
        prob_data = [["Emotion", "Probability", "Bar"]]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for em, p in sorted_probs:
            bar = "█" * int(p * 20)
            prob_data.append([em.capitalize(), f"{p:.4f} ({p*100:.1f}%)", bar])

        t2 = Table(prob_data, colWidths=[5*cm, 5*cm, 7*cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), green),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F0EDE6")]),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#D4CFC7")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.5*cm))

    # Model metrics
    if eval_metrics:
        story.append(Paragraph("Model Performance Metrics", h2_style))
        metric_data = [
            ["Metric", "Score"],
            ["Accuracy", f"{eval_metrics.get('accuracy', 0):.4f}"],
            ["Precision (Macro)", f"{eval_metrics.get('precision_macro', 0):.4f}"],
            ["Recall (Macro)", f"{eval_metrics.get('recall_macro', 0):.4f}"],
            ["F1 Score (Macro)", f"{eval_metrics.get('f1_macro', 0):.4f}"],
        ]
        t3 = Table(metric_data, colWidths=[8*cm, 8*cm])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), green),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F0EDE6")]),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#D4CFC7")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t3)

    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=rl_colors.HexColor("#D4CFC7")))
    story.append(Paragraph(
        "Generated by AI-Powered Speech Emotion Recognition Platform | Confidential",
        ParagraphStyle("Footer", fontSize=8, textColor=rl_colors.HexColor("#718096"),
                       alignment=1)
    ))

    doc.build(story)
    return buf.getvalue()


def _generate_text_pdf(analysis_result: Dict) -> bytes:
    """Plain-text fallback when reportlab is unavailable."""
    lines = [
        "AI-Powered Speech Emotion Recognition - Analysis Report",
        "=" * 60,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Predicted Emotion: {analysis_result.get('emotion', 'N/A')}",
        f"Confidence: {analysis_result.get('confidence', 0):.2%}",
        f"Model: {analysis_result.get('model_name', 'SER Model')}",
        "",
        "Emotion Probabilities:",
    ]
    for e, p in analysis_result.get("probabilities", {}).items():
        lines.append(f"  {e.capitalize()}: {p:.4f}")

    return "\n".join(lines).encode()
