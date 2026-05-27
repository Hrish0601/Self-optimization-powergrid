# generate_load_csv.py
import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 1000

data = {
    'load_demand_kwh': np.random.uniform(10, 150, n),
    'transformer_load_percent': np.random.uniform(50, 100, n),
    'hour_of_day': np.random.randint(0, 24, n),
    'temperature_c': np.random.uniform(20, 40, n),
    'humidity_percent': np.random.uniform(40, 90, n),
    'renewable_contribution_percent': np.random.uniform(0, 30, n),
    'frequency_hz': np.random.uniform(49.9, 50.1, n),
    'season_code': np.random.choice([0, 1, 2], n)  # 0=Summer, 1=Monsoon, 2=Winter
}

df = pd.DataFrame(data)
os.makedirs('load_bal', exist_ok=True)
df.to_csv('load_bal/syn_load_bal.csv', index=False)
print("syn_load_bal.csv created with season_code!")