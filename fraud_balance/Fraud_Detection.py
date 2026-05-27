# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("C:\\Users\\hrish\\Downloads\\Project\\Project\\fraud_balance\\syn_fraud.csv")

# =========================
# CREATE FRAUD LABEL
# =========================
data['fraudulent'] = 0

fraud_condition = (
    (data['load_demand_kwh'] > np.percentile(data['load_demand_kwh'], 85)) &
    (data['deviation_neighborhood_ratio'] > np.percentile(data['deviation_neighborhood_ratio'], 80))
)

data.loc[fraud_condition, 'fraudulent'] = 1

print("\nFraud Distribution:")
print(data['fraudulent'].value_counts())

# =========================
# BALANCE DATA
# =========================
fraud = data[data['fraudulent'] == 1]
normal = data[data['fraudulent'] == 0]

normal_sampled = normal.sample(len(fraud) * 2, random_state=42)
data = pd.concat([fraud, normal_sampled]).reset_index(drop=True)

print("\nBalanced Distribution:")
print(data['fraudulent'].value_counts())

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

# =========================
# ENCODING
# =========================
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# =========================
# PREPARE DATA
# =========================
X = data[numerical_cols + categorical_cols].copy()
y = data['fraudulent']

X[numerical_cols] = X[numerical_cols].astype(float)

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\n=== MODEL PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# SAVE
# =========================
os.makedirs('static', exist_ok=True)

joblib.dump(model, 'static/fraud_model_local.joblib')
joblib.dump(scaler, 'static/scaler.joblib')
joblib.dump(label_encoders, 'static/encoders.joblib')

print("✅ Model saved")