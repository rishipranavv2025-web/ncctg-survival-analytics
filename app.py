import io
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

import os
from config import FEATURES, DISPLAY_NAMES, MODEL_PATH
from survival_models import ExponentialPHModel


app = Flask(__name__)

required_files = [
    MODEL_PATH / "cox.pkl",
    MODEL_PATH / "weibull.pkl",
    MODEL_PATH / "exponential.pkl",
    MODEL_PATH / "metadata.json"
]
missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "Trained model files are missing. Run `py train.py` first. Missing: "
        + ", ".join(missing_files)
    )

cox = joblib.load(MODEL_PATH / "cox.pkl")
weibull = joblib.load(MODEL_PATH / "weibull.pkl")
exp_model = joblib.load(MODEL_PATH / "exponential.pkl")
with open(MODEL_PATH / "metadata.json", encoding="utf-8") as file:
    metadata = json.load(file)

selected_model = metadata.get("selected_model", "Cox PH")
valid_models = {"Cox PH", "Weibull AFT", "Exponential PH"}
if selected_model not in valid_models:
    raise ValueError(f"Unsupported selected model in metadata: {selected_model}")


def make_input(form):
    """Validate form values and map sex to the exact training encoding."""
    try:
        age = float(form["age"])
        ecog = float(form["ph_ecog"])
        ph_karno = float(form["ph_karno"])
        pat_karno = float(form["pat_karno"])
        meal_cal = float(form["meal_cal"])
        wt_loss = float(form["wt_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Enter a valid number for every patient field.") from exc

    sex_text = str(form.get("sex", "")).strip().lower()
    if sex_text in {"male", "m", "0"}:
        sex_code = 0
        sex_label = "Male"
    elif sex_text in {"female", "f", "1"}:
        sex_code = 1
        sex_label = "Female"
    else:
        raise ValueError("Choose Male or Female.")

    ranges = [
        ("Age", age, 18, 110),
        ("ECOG score", ecog, 0, 4),
        ("Physician Karnofsky score", ph_karno, 0, 100),
        ("Patient Karnofsky score", pat_karno, 0, 100),
        ("Meal calories", meal_cal, 0, 10000),
        ("Weight loss", wt_loss, -100, 300),
    ]
    for label, value, low, high in ranges:
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{label} must be between {low} and {high}.")

    values = {
        "age": age,
        "sex": sex_code,
        "ph.ecog": ecog,
        "ph.karno": ph_karno,
        "pat.karno": pat_karno,
        "meal.cal": meal_cal,
        "wt.loss": wt_loss,
    }
    patient = pd.DataFrame([values], columns=FEATURES)

    # The existing template needs numeric sex for its hidden PDF form and
    # a readable label for the visible select. See the two template edits below.
    form_values = dict(values)
    form_values["sex_label"] = sex_label
    return patient, form_values


def survival_at_days(model_name, patient, days):
    days = float(days)
    if days < 0 or not np.isfinite(days):
        raise ValueError("Prediction time must be a finite non-negative number.")

    if model_name == "Cox PH":
        value = cox.predict_survival_function(patient[FEATURES], times=[days]).iloc[0, 0]
    elif model_name == "Weibull AFT":
        value = weibull.predict_survival_function(patient[FEATURES], times=[days]).iloc[0, 0]
    elif model_name == "Exponential PH":
        hazard = float(exp_model.predict_hazard(patient[FEATURES].to_numpy(dtype=float))[0])
        value = np.exp(-hazard * days)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return float(np.clip(value, 0.0, 1.0))


def make_curve(model_name, patient):
    times = np.linspace(0, 1095, 180)
    if model_name == "Cox PH":
        survival = cox.predict_survival_function(
            patient[FEATURES], times=times
        ).iloc[:, 0].to_numpy()
    elif model_name == "Weibull AFT":
        survival = weibull.predict_survival_function(
            patient[FEATURES], times=times
        ).iloc[:, 0].to_numpy()
    elif model_name == "Exponential PH":
        hazard = float(exp_model.predict_hazard(patient[FEATURES].to_numpy(dtype=float))[0])
        survival = np.exp(-hazard * times)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    survival = np.clip(np.asarray(survival, dtype=float), 0.0, 1.0)
    return [
        {"x": round(float(t), 1), "y": round(float(s), 5)}
        for t, s in zip(times, survival)
    ]


def risk_category(p365):
    # Project-specific display categories; not validated clinical thresholds.
    if p365 >= 0.70:
        return "Lower estimated risk band"
    if p365 >= 0.40:
        return "Intermediate estimated risk band"
    return "Higher estimated risk band"


def build_report(result, patient_values):
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.6 * cm, leftMargin=1.6 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="SmallNote", parent=styles["BodyText"], fontSize=8.5,
        leading=11, textColor=colors.HexColor("#475569")
    ))
    story = [
        Paragraph("Personalized Lung Cancer Survival — Project Report", styles["Title"]),
        Paragraph(datetime.now().strftime("Generated %d %B %Y, %H:%M"), styles["SmallNote"]),
        Spacer(1, 10),
        Paragraph("Selected model", styles["Heading2"]),
        Paragraph(result["model"], styles["BodyText"]),
        Spacer(1, 10),
    ]

    patient_table = [["Input", "Value"]]
    for feature in FEATURES:
        value = patient_values[feature]
        if feature == "sex":
            value = "Male" if int(value) == 0 else "Female"
        elif isinstance(value, (float, np.floating)):
            value = f"{float(value):g}"
        patient_table.append([DISPLAY_NAMES.get(feature, feature), str(value)])

    table = Table(patient_table, colWidths=[8 * cm, 8 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([table, Spacer(1, 12), Paragraph("Estimated survival", styles["Heading2"])])

    estimate_table = [["Horizon", "Estimated survival"]]
    estimate_table.extend([
        [label, f"{value:.1f}%"]
        for label, value in result["probabilities"].items()
    ])
    table = Table(estimate_table, colWidths=[8 * cm, 8 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        table, Spacer(1, 12),
        Paragraph(f"Project presentation band: <b>{result['risk']}</b>", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(
            "Important: this is an academic/research prototype using a historical "
            "survival dataset. The displayed probabilities and presentation band "
            "are not clinically validated and must not be used for medical decisions.",
            styles["SmallNote"
        ])
    ])
    document.build(story)
    buffer.seek(0)
    return buffer


def make_result(patient, patient_values):
    # The model is selected once during training-only CV and frozen in metadata.
    horizons_days = {"1 year": 365, "2 years": 730, "3 years": 1095}
    raw_probabilities = {
        label: survival_at_days(selected_model, patient, days)
        for label, days in horizons_days.items()
    }
    return {
        "model": selected_model,
        "probabilities": {
            label: round(probability * 100, 1)
            for label, probability in raw_probabilities.items()
        },
        "risk": risk_category(raw_probabilities["1 year"]),
        "curve": make_curve(selected_model, patient),
        "patient": patient_values,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {}

    if request.method == "POST":
        # Preserve entered fields even when validation fails.
        form_values = {
            "age": request.form.get("age", ""),
            "sex": request.form.get("sex", "Male"),
            "ph.ecog": request.form.get("ph_ecog", ""),
            "ph.karno": request.form.get("ph_karno", ""),
            "pat.karno": request.form.get("pat_karno", ""),
            "meal.cal": request.form.get("meal_cal", ""),
            "wt.loss": request.form.get("wt_loss", ""),
        }
        try:
            patient, numeric_values = make_input(request.form)
            form_values.update(numeric_values)
            form_values["sex_label"] = numeric_values["sex_label"]
            result = make_result(patient, numeric_values)
        except Exception as exc:
            error = f"Please check the inputs. {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        metrics=metadata.get("metrics", {}),
        selected_model=selected_model,
        rows=metadata.get("rows", "—"),
        events=metadata.get("events", "—"),
        censored=metadata.get("censored", "—"),
        train_rows=metadata.get("train_rows", "—"),
        test_rows=metadata.get("test_rows", "—"),
        form_values=form_values,
        display_names=DISPLAY_NAMES,
    )


@app.post("/download-report")
def download_report():
    try:
        patient, values = make_input(request.form)
        result = make_result(patient, values)
        pdf = build_report(result, values)
        return send_file(
            pdf, as_attachment=True,
            download_name="lung_survival_prediction_report.pdf",
            mimetype="application/pdf"
        )
    except Exception as exc:
        return (
            "Could not generate the report. Please return to the form and check "
            f"the entered values. Details: {exc}",
            400
        )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "selected_model": selected_model,
        "selection_method": metadata.get("selection_method", "Not recorded"),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    app.run(host=host, port=port, debug=False)