import threading
import queue
import time
import pandas as pd
from src.decision.alert_generator import generate_alerts

def replay_data(df, q, config):
    batch_size = config['replay']['batch_interval']
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        alerts = generate_alerts(batch, config)
        q.put(alerts)
        time.sleep(batch_size * config['replay']['speed'])