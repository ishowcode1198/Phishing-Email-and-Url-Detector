import pickle
from pathlib import Path

from feature_extraction import get_rule_based_warnings, summarize_confidence


def load_model(model_path="model.pkl"):
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_model()


def predict_input(text: str):
    prompt = text.strip()
    if not prompt:
        return {
            "label": "safe",
            "confidence": 0.0,
            "confidence_text": "No input provided.",
            "warnings": ["Please enter email text or a URL to analyze."],
        }

    prediction = model.predict([prompt])[0]
    if prediction == "phishing":
        prediction = "not_safe"

    probabilities = model.predict_proba([prompt])[0]
    confidence_score = float(max(probabilities))
    warnings = get_rule_based_warnings(prompt)

    if prediction == "not_safe" and warnings:
        confidence_score = max(confidence_score, min(0.92, 0.60 + 0.08 * len(warnings)))
    if prediction == "safe" and not warnings:
        confidence_score = max(confidence_score, 0.65)

    return {
        "label": prediction,
        "confidence": confidence_score,
        "confidence_text": summarize_confidence(confidence_score),
        "warnings": warnings,
    }
