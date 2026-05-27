# load_bal/load_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class LoadBalancingEnv(gym.Env):
    def __init__(self, csv_path=None):
        super().__init__()
        
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            required = [
                'load_demand_kwh', 'transformer_load_percent', 'hour_of_day',
                'temperature_c', 'humidity_percent', 'renewable_contribution_percent',
                'frequency_hz', 'season_code'
            ]
            df = df[required]
            self.data = df.to_dict('records')
        else:
            self.data = [{
                "load_demand_kwh": 20, "transformer_load_percent": 70, "hour_of_day": 12,
                "temperature_c": 28, "humidity_percent": 60, "renewable_contribution_percent": 15,
                "frequency_hz": 0.5, "season_code": 0  # frequency_hz normalized
            }] * 100
        
        self.current_step = 0
        self.max_steps = len(self.data)

        # MATCHES YOUR TRAINED MODEL'S BOUNDS
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([18.4, 150.0, 23.0, 50.0, 100.0, 30.0, 1.0, 2.0]),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        r = self.data[self.current_step]
        return np.array([
            r["load_demand_kwh"] / 8,  # 0–18.4
            r["transformer_load_percent"],
            r["hour_of_day"],
            r["temperature_c"],
            r["humidity_percent"],
            r["renewable_contribution_percent"],
            (r["frequency_hz"] - 49.0) / 2.0,  # 49–51 → 0–1
            r["season_code"]
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        load = self.data[self.current_step - 1]["load_demand_kwh"]
        adjusted = load + action[0] * 10  # scale action
        reward = -abs(adjusted - 25) - abs(action[0]) * 0.1
        info = {
            "adjusted_load": adjusted,
            "transformer_load_percent": self.data[self.current_step - 1]["transformer_load_percent"],
            "overload_count": 1 if adjusted > 40 else 0
        }
        return self._get_obs(), reward, done, False, info

    def render(self):
        pass