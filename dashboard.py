# dashboard.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import json
import os

print("Loading datasets...")

load_df = pd.read_csv('load_bal/syn_load_bal.csv')
fraud_df = pd.read_csv('fraud_balance/syn_fraud.csv')

# =========================
# ZONES WITH INTELLIGENCE
# =========================
zones = {
    'BEST Central':   {'lat': 19.0760, 'lng': 72.8777, 'risk_factor': 0.3},
    'Adani West':     {'lat': 19.1132, 'lng': 72.8369, 'risk_factor': 0.6},
    'Tata South':     {'lat': 19.0618, 'lng': 72.8350, 'risk_factor': 0.4},
    'MSEDCL Suburbs': {'lat': 19.2183, 'lng': 72.9781, 'risk_factor': 0.7}
}

zone_explanations = {
    'BEST Central': {
        "risk_level": "Low",
        "reason": "Stable infrastructure and balanced demand",
        "advice": "Normal usage is safe"
    },
    'Adani West': {
        "risk_level": "Medium",
        "reason": "Moderate evening load spikes",
        "advice": "Avoid peak-hour usage"
    },
    'Tata South': {
        "risk_level": "Low-Medium",
        "reason": "Residential area with occasional spikes",
        "advice": "Monitor heavy appliance usage"
    },
    'MSEDCL Suburbs': {
        "risk_level": "High",
        "reason": "Frequent overload and unstable grid behavior",
        "advice": "Shift usage to off-peak hours"
    }
}

# =========================
# ASSIGN ZONES
# =========================
fraud_df['zone'] = np.random.choice(list(zones.keys()), len(fraud_df))

for col in ['lat', 'lng', 'risk_factor']:
    fraud_df[col] = fraud_df['zone'].map(lambda z: zones[z][col])

# =========================
# ADD EXPLANATIONS
# =========================
fraud_df['zone_risk_level'] = fraud_df['zone'].map(lambda z: zone_explanations[z]["risk_level"])
fraud_df['zone_reason'] = fraud_df['zone'].map(lambda z: zone_explanations[z]["reason"])

# =========================
# IMPROVED BLACKOUT SCORE
# =========================
fraud_df['blackout_score'] = (
    fraud_df['load_spike_flag'].astype(float) * 0.6 +
    fraud_df['deviation_neighborhood_ratio'] * fraud_df['risk_factor'] +
    (fraud_df['load_demand_kwh'] / fraud_df['load_demand_kwh'].max()) * 0.4
)

fraud_df['blackout_risk'] = (fraud_df['blackout_score'] > 1.2).astype(int)

# =========================
# FEATURES
# =========================
features = ['load_demand_kwh', 'deviation_neighborhood_ratio']

X = fraud_df[features]
y = fraud_df['blackout_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=80, random_state=42)
model.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================
os.makedirs("static", exist_ok=True)
dump(model, "static/blackout_model.joblib")

# =========================
# GEOJSON EXPORT
# =========================

geojson = {"type": "FeatureCollection", "features": []}

for name, info in zones.items():
    lat, lng = info["lat"], info["lng"]

    geojson["features"].append({
        "type": "Feature",
        "properties": {
            "name": name,
            "risk_factor": info["risk_factor"],
            "risk_level": zone_explanations[name]["risk_level"],
            "reason": zone_explanations[name]["reason"],
            "advice": zone_explanations[name]["advice"]
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lng-0.015, lat-0.015],
                [lng+0.015, lat-0.015],
                [lng+0.015, lat+0.015],
                [lng-0.015, lat+0.015],
                [lng-0.015, lat-0.015]
            ]]
        }
    })
    
with open("static/mumbai_zones.geojson", "w") as f:
    json.dump(geojson, f, indent=2)

print("✅ GeoJSON saved -> static/mumbai_zones.geojson")

# =========================
# FRONTEND DATA FILE
# =========================
with open("static/zone_info.json", "w") as f:
    json.dump({
        z: {
            **zones[z],
            **zone_explanations[z]
        } for z in zones
    }, f, indent=2)

print("✅ Dashboard fully generated")