
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = ["voltage", "current", "temperature", "vibration", "power_factor"]
feature_map = {
    "voltage": "abnormal voltage",
    "current": "abnormal current",
    "temperature": "excessive temperature",
    "vibration": "high vibration",
    "power_factor": "low power factor"
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[f]) for f in feature_names]
        scaled = scaler.transform([data])
        prediction = model.predict(scaled)[0]

        contributions = scaled[0] * model.feature_importances_
        top_idx = int(np.argmax(contributions))
        top_feature = feature_names[top_idx]
        reason = feature_map.get(top_feature, "unknown issue")

        if prediction == 1:
            message = f"⚠️ Maintenance Alert: Likely cause is **{reason}**."
        else:
            message = "✅ All systems are operating normally."

        return render_template("index.html", result=message)
    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
