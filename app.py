# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
app.secret_key = "mumbai-grid-2025-secure-key"
bcrypt = Bcrypt(app)

# =========================
# DATABASE
# =========================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///powergrid.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# =========================
# LOAD MODEL
# =========================
try:
    fraud_model = joblib.load('C:\\Users\\hrish\\Downloads\\Project\\Project\\static\\fraud_model_local.joblib')
    scaler = joblib.load('C:\\Users\\hrish\\Downloads\\Project\\Project\\static\\scaler.joblib')
    encoders = joblib.load('C:\\Users\\hrish\\Downloads\\Project\\Project\\static\\encoders.joblib')
except:
    fraud_model = None
    scaler = None
    encoders = None
    print("⚠️ Fraud model not loaded — using fallback")

# =========================
# FEATURES
# =========================
numerical_cols = [
    'voltage_v','current_a','active_power_kw',
    'reactive_power_kvar','load_demand_kwh',
    'occupancy_level','deviation_neighborhood_ratio'
]

categorical_cols = [
    'household_type','appliance_usage_category',
    'broader_zone','locality'
]

ZONES = ['BEST Central', 'Adani West', 'Tata South', 'MSEDCL Suburbs']

# =========================
# FRAUD FUNCTION (UPDATED)
# =========================
def simple_fraud_check(data):

    feature_order = numerical_cols + categorical_cols

    df = pd.DataFrame([{col: data.get(col) for col in feature_order}])

    df = df.fillna({
        'occupancy_level': 2,
        'household_type': 'Apartment',
        'appliance_usage_category': 'Medium',
        'broader_zone': 'Urban',
        'locality': 'Mumbai'
    })

    # Safe encoding
    for col in categorical_cols:
        try:
            df[col] = encoders[col].transform(df[col])
        except:
            df[col] = 0

    # Scale
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Predict
    fraud_prob = fraud_model.predict_proba(df)[0][1]

    print("Fraud Probability:", fraud_prob)

    # =========================
    # EXPLANATION SYSTEM
    # =========================
    def generate_explanation(data):
        reasons = []

        if data.get('load_demand_kwh', 0) > 80:
            reasons.append("High energy consumption")

        if data.get('deviation_neighborhood_ratio', 0) > 2:
            reasons.append("Abnormal usage compared to neighborhood")

        if data.get('current_a', 0) > 18:
            reasons.append("Unusually high current flow")

        if data.get('active_power_kw', 0) > 5:
            reasons.append("High power usage")

        if data.get('load_spike_flag', 0) == 1:
            reasons.append("Sudden load spike detected")

        if not reasons:
            reasons.append("Usage within normal range")

        return reasons

    explanation = generate_explanation(data)

    return {
        "fraud_prob": round(float(fraud_prob), 3),
        "label": "Fraud Detected" if fraud_prob > 0.4 else "Normal Usage",
        "confidence": round(0.6 + abs(fraud_prob - 0.5), 2),
        "explanation": explanation
    }

# =========================
# LOAD BALANCING (UNCHANGED)
# =========================
def optimize_for_home(params):
    # Extract correct inputs from frontend
    demand = params.get('total_demand_mw', 50)
    capacity = params.get('available_capacity_mw', 80)
    peak = params.get('peak_hour_factor', 1.0)
    renewable = params.get('renewable_share_percent', 20)
    temp = params.get('temperature_c', 25)
    event = params.get('unexpected_event_flag', 0)
    time = params.get('time_of_day', 'day')

    # ---------------------------
    # CORE CALCULATION
    # ---------------------------
    effective_load = demand * peak

    stress_ratio = effective_load / capacity

    # temperature impact
    if temp > 35:
        stress_ratio += 0.1

    # renewable benefit
    stress_ratio -= renewable / 200

    # unexpected event penalty
    if event == 1:
        stress_ratio += 0.15

    # ---------------------------
    # DECISION LOGIC
    # ---------------------------
    if stress_ratio > 0.9:
        action = -0.4
        recommendation = "Urgent load reduction required. Shift heavy appliances immediately."
        risk = "High"

    elif stress_ratio > 0.7:
        action = -0.2
        recommendation = "Reduce usage during peak hours."
        risk = "Medium"

    else:
        action = 0.1
        recommendation = "System stable. You can continue normal usage."
        risk = "Low"

    adjusted_load = max(0, effective_load + action * 20)

    savings = max(0, (effective_load - adjusted_load) * 5)

    return {
        "Suggested_Action": round(action, 2),
        "Adjusted_Load_MW": round(adjusted_load, 1),   # ✅ changed
        "Saving_Estimate_Rs": round(savings, 0),
        "Recommendation": recommendation,
        "Overload_Risk": risk
     }

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

# ✅ UPDATED FRAUD ROUTE
@app.route('/fraud-detection', methods=['GET', 'POST'])
def fraud_detection():
    if request.method == 'POST':
        data = request.get_json()
        result = simple_fraud_check(data)

        return jsonify({
            "status": "success",
            "fraud_prob": result["fraud_prob"],
            "label": result["label"],
            "confidence": result["confidence"],
            "explanation": result["explanation"]
        })

    return render_template('fraud_detection.html')

@app.route('/load-balancing', methods=['GET', 'POST'])
def load_balancing():
    if request.method == 'POST':
        data = request.get_json()
        result = optimize_for_home(data)
        return jsonify(result)
    return render_template('load_balancing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/renewables')
def renewables():
    try:
        df = pd.read_csv('load_bal/syn_load_bal.csv')
        if 'region' not in df.columns:
            df['region'] = [ZONES[i % 4] for i in range(len(df))]
        
        zone_means = df.groupby('region')['renewable_contribution_percent'].mean().round(1).tolist()
        global_avg = round(df['renewable_contribution_percent'].mean(), 1)
    except:
        zone_means = [18.5, 24.1, 19.8, 14.2]
        global_avg = 19.2

    return render_template('renewables.html', 
                         zone_labels=ZONES,
                         zone_values=zone_means,
                         global_avg=global_avg)

@app.route('/forecast')
def forecast():
    try:
        df = pd.read_csv('load_bal/syn_load_bal.csv')
        predictions = (df.tail(24)['load_demand_kwh'] / 80).round(3).tolist()
    except:
        predictions = [0.45] * 24
    return render_template('forecast.html', predictions=predictions)

@app.route('/alerts')
def alerts():
    alerts = [
        {"zone": "MSEDCL Suburbs", "risk": "High", "message": "High chance of load shedding tonight", "time": "Just now"},
        {"zone": "Adani West", "risk": "Medium", "message": "Peak load time - shift heavy appliances", "time": "1 hour ago"},
    ]
    return render_template('alerts.html', alerts=alerts)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = "User"
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/professional')
def professional():
    return render_template('professional.html')

@app.route('/fraud-detection-pro')
def fraud_detection_pro():
    return render_template('fraud_detection_pro.html')

@app.route('/load-balancing-pro')
def load_balancing_pro():
    return render_template('load_balancing_pro.html')


@app.route('/api/load-balancing-pro', methods=['POST'])
def load_balancing_pro_api():
    data = request.get_json()

    print("INPUT:", data)

    result = optimize_for_home(data)

    print("OUTPUT:", result)

    return jsonify(result)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("🚀 Mumbai Smart Energy App Running...")
    app.run(debug=True, host='0.0.0.0', port=5000)