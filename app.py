from sklearn.naive_bayes import GaussianNB
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import tldextract
import re

app = Flask(__name__)
CORS(app)

# --- Load the trained model ---
try:
    model = joblib.load("rf_model.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print("⚠️ Could not load model:", e)
    model = None
nb_model=GaussianNB()

# --- Feature extraction function ---
def extract_features(url):
    ext = tldextract.extract(url)
    length = len(url)
    has_at = 1 if "@" in url else 0
    dots = url.count(".")
    suspicious = 1 if re.search(r"(login|secure|account|bank|verify)", url, re.IGNORECASE) else 0
    return [length, has_at, dots, suspicious]

# --- TEST ROUTE (VERY IMPORTANT) ---
@app.route("/", methods=["GET"])
def home():
    return "Backend is running successfully 🚀"

# --- API endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    print("📩 Request received")

    data = request.get_json() or {}
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    features = extract_features(url)

    if model is None:
        return jsonify({
            "url": url,
            "label": "unknown",
            "confidence": 0.0
        })

    # --- Random Forest Prediction ---
    rf_proba = model.predict_proba([features])[0]
    rf_idx = rf_proba.argmax()
    rf_label = model.classes_[rf_idx]
    rf_conf = float(rf_proba[rf_idx])

    # --- Naive Bayes (dummy training for demo) ---
    X_train = [
        [20,0,2,0],
        [80,1,5,1],
        [60,0,3,1],
        [25,0,2,0]
    ]
    y_train = ["safe", "phishing", "phishing", "safe"]

    nb_model.fit(X_train, y_train)

    nb_pred = nb_model.predict([features])[0]

    return jsonify({
        "url": url,
        "rf_label": str(rf_label),
        "rf_confidence": round(rf_conf, 3),
        "nb_prediction": str(nb_pred)
    })

# --- RUN SERVER ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 
