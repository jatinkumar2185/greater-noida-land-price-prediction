"""
=============================================================
app.py  —  Greater Noida Land Price Prediction
B.Tech Major Project  |  Flask Web Application
=============================================================
ROUTES:
  GET  /          → Show prediction form  (index.html)
  POST /predict   → Show prediction result (result.html)
  GET  /metrics   → JSON endpoint with model metrics
=============================================================
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import os
import numpy as np
import pandas as pd

# ── App Setup ──────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load Model & Support Files ─────────────────────────────
model       = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
feature_cols = json.load(open(os.path.join(BASE_DIR, "feature_columns.json")))
sector_map   = json.load(open(os.path.join(BASE_DIR, "sector_map.json")))
metrics      = json.load(open(os.path.join(BASE_DIR, "metrics.json")))

# Full sector list for dropdown (sorted by prestige)
SECTORS = [
    "Pari Chowk", "Alpha 1", "Alpha 2", "Omicron 1", "Beta 1",
    "Omicron 2", "Beta 2", "Knowledge Park 1", "Knowledge Park 2",
    "Gamma 1", "Zeta 1", "Gamma 2", "Zeta 2", "Delta 1", "Delta 2",
    "Sector 1", "Sector 2", "Techzone IV", "Sector 3", "Ecotech 1",
]


def format_indian_currency(amount: float) -> str:
    """
    Format a number in Indian currency style.
    Example: 12345678 → ₹1,23,45,678
    Also returns a human-readable label (Lakhs / Crores).
    """
    amount = int(round(amount, -3))  # Round to nearest ₹1000

    # Indian comma formatting
    s = str(amount)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest   = s[:-3]
        parts  = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        parts.reverse()
        formatted = ",".join(parts) + "," + last3

    # Human label
    if amount >= 1_00_00_000:   # ≥ 1 Crore
        label = f"₹{amount/1_00_00_000:.2f} Crore"
    elif amount >= 1_00_000:    # ≥ 1 Lakh
        label = f"₹{amount/1_00_000:.2f} Lakh"
    else:
        label = f"₹{formatted}"

    return f"₹{formatted}", label


def build_feature_vector(form_data: dict) -> np.ndarray:
    """
    Convert raw form data into a feature vector matching
    the exact columns the trained model expects.
    """
    sector       = form_data["sector"]
    area         = float(form_data["area"])
    road_width   = int(form_data["road_width"])
    metro_dist   = float(form_data["metro_dist"])
    airport_dist = float(form_data["airport_dist"])
    corner_plot  = 1 if form_data.get("corner_plot") == "Yes" else 0
    facing       = form_data["facing"]
    nearby_school = 1 if form_data.get("nearby_school") == "Yes" else 0
    nearby_hosp   = 1 if form_data.get("nearby_hospital") == "Yes" else 0
    comm_nearby   = 1 if form_data.get("commercial_nearby") == "Yes" else 0

    # Sector code
    sector_code = sector_map.get(sector, 10)

    # Facing one-hot (East was dropped as reference class)
    facing_north = 1 if facing == "North" else 0
    facing_south = 1 if facing == "South" else 0
    facing_west  = 1 if facing == "West"  else 0

    # Build feature dict matching training columns exactly
    feat = {
        "Area_sqm":          area,
        "Road_Width_ft":     road_width,
        "Metro_Dist_km":     metro_dist,
        "Airport_Dist_km":   airport_dist,
        "Corner_Plot":       corner_plot,
        "Nearby_School":     nearby_school,
        "Nearby_Hospital":   nearby_hosp,
        "Commercial_Nearby": comm_nearby,
        "Facing_North":      facing_north,
        "Facing_South":      facing_south,
        "Facing_West":       facing_west,
        "Sector_Code":       sector_code,
    }

    # Return as array in correct column order
    import pandas as pd
    return pd.DataFrame([feat])[feature_cols]


# ══════════════════════════════════════════════════════════
# ROUTE 1:  GET /  →  Show the prediction form
# ══════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", sectors=SECTORS, metrics=metrics)


# ══════════════════════════════════════════════════════════
# ROUTE 2:  POST /predict  →  Make prediction & show result
# ══════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # Validate required fields
        required = ["sector", "area", "road_width", "metro_dist", "airport_dist", "facing"]
        for field in required:
            if not form.get(field):
                return render_template("index.html", sectors=SECTORS, metrics=metrics,
                                       error=f"Field '{field}' is required.")

        # Build feature vector
        X = build_feature_vector(form)

        # Predict
        predicted_price = float(model.predict(X)[0])
        predicted_price = max(500_000, predicted_price)  # floor ₹5 lakh

        # Format price
        formatted_price, price_label = format_indian_currency(predicted_price)

        # Price per sqm
        area = float(form["area"])
        price_per_sqm = int(predicted_price / area)
        psm_fmt, psm_label = format_indian_currency(price_per_sqm)

        # Collect input for display on result page
        input_data = {
            "Sector":             form["sector"],
            "Area":               f"{area:.0f} sqm",
            "Road Width":         f"{form['road_width']} ft",
            "Metro Distance":     f"{form['metro_dist']} km",
            "Airport Distance":   f"{form['airport_dist']} km",
            "Corner Plot":        form.get("corner_plot", "No"),
            "Facing":             form["facing"],
            "Nearby School":      form.get("nearby_school", "No"),
            "Nearby Hospital":    form.get("nearby_hospital", "No"),
            "Commercial Nearby":  form.get("commercial_nearby", "No"),
        }

        return render_template(
            "result.html",
            formatted_price=formatted_price,
            price_label=price_label,
            price_per_sqm=f"₹{price_per_sqm:,}",
            input_data=input_data,
            metrics=metrics,
        )

    except Exception as e:
        return render_template("index.html", sectors=SECTORS, metrics=metrics,
                               error=f"Prediction error: {str(e)}")


# ══════════════════════════════════════════════════════════
# ROUTE 3:  GET /metrics  →  JSON model metrics
# ══════════════════════════════════════════════════════════
@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics)


# ── Entry Point ────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Land Price Prediction App starting...")
    print("   Open: http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
