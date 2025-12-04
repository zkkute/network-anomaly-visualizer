import time
import json
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
import numpy as np
from src.ingestion.live_capture import start_live_capture, packet_queue
from src.preprocessing.flow_aggregator import aggregate_flows
from src.preprocessing.feature_extractor import extract_features
from src.ml.inference import get_hybrid_score
from src.decision.alert_generator import generate_alerts

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.live_capture import start_live_capture, packet_queue
from src.preprocessing.flow_aggregator import aggregate_flows
...

import yaml

CONFIG_PATH = "config/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

OUTPUT_FLOWS = Path("data/live/flows_live.csv")
OUTPUT_ALERTS = Path("data/live/alerts.json")

def realtime_engine():
    print("[ENGINE] Запуск real-time анализа...")
    start_live_capture(config["live"]["interface"])

    buffer = []
    last_process = time.time()

    OUTPUT_FLOWS.parent.mkdir(exist_ok=True)

    while True:
        # 1) читаем пакеты из очереди
        try:
            pkt = packet_queue.get(timeout=0.2)
            buffer.append(pkt)
        except:
            pass

        # 2) каждые N секунд: обработка
        if time.time() - last_process >= config["live"]["interval"]:
            if len(buffer) == 0:
                continue

            print(f"[ENGINE] Обработка {len(buffer)} пакетов...")

            df = pd.DataFrame(buffer)
            buffer.clear()

            # Агрегация flows
            flows = aggregate_flows(df)

            # Генерация 82 признаков
            features_df = extract_features(flows)

            # ML score
            dummy_scaler = StandardScaler()
            dummy_scaler.mean_ = np.zeros(7)
            dummy_scaler.scale_ = np.ones(7)

            X = dummy_scaler.transform(
                features_df[['pps', 'bps', 'duration', 'packets', 'bytes', 'syn_ratio', 'port_entropy']])
            scores = np.random.random(len(X))  # или заглушка любаяes

            # Alerts (heatmap)
            alerts_df = generate_alerts(features_df, config)

            # Сохранение данных
            features_df.to_csv(OUTPUT_FLOWS, index=False)
            alerts_df.to_json(OUTPUT_ALERTS, orient="records")

            print(f"[ENGINE] OK: обновлены файлы {OUTPUT_FLOWS} и {OUTPUT_ALERTS}")

            last_process = time.time()


if __name__ == "__main__":
    realtime_engine()
