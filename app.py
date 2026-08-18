"""
app.py
------
Flask web app + REST API serving the trained churn model.

Run:
    python app.py

Then open http://127.0.0.1:5000 in a browser, or POST JSON to
http://127.0.0.1:5000/api/predict
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from flask import Flask, request, jsonify, render_template
from predict import predict_churn
from preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES

app = Flask(__name__)

REQUIRED_FIELDS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        result = predict_churn(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
