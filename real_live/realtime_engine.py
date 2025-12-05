# real_live/realtime_engine.py

import time
import pandas as pd
from pathlib import Path
import sys
import joblib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.live_capture import start_live_capture, packet_queue
from src.preprocessing.flow_aggregator import aggregate_flows
from src.preprocessing.feature_extractor import extract_features

# ---------------------- CONFIG & PATHS ----------------------
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

OUTPUT_FLOWS = Path("data/live/flows_live.csv")
OUTPUT_ALERTS = Path("data/live/alerts.json")
OUTPUT_FLOWS.parent.mkdir(exist_ok=True)

# ---------------------- LOAD MODELS ----------------------
FEATURES = joblib.load("models/feature_list.pkl")
scaler = joblib.load("models/scaler.pkl")
iso_model = joblib.load("models/isolation_forest.pkl")

try:
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_model.json")
    HAS_XGB = True
    print("[ML] XGBoost загружен")
except:
    HAS_XGB = False
    print("[ML] XGBoost не найден → только Isolation Forest")

# ---------------------- HYBRID SCORE ----------------------
def get_hybrid_score(numeric_df):
    """Принимает DataFrame ТОЛЬКО с 45 числовыми признаками"""
    X = scaler.transform(numeric_df)
    iso_anom = -iso_model.decision_function(X)
    iso_score = (iso_anom - iso_anom.min()) / (iso_anom.ptp() + 1e-12)

    if HAS_XGB:
        xgb_proba = xgb_model.predict_proba(X)[:, 1]
        return np.maximum(xgb_proba, iso_score)
    return iso_score

# ---------------------- ALERTS ----------------------
def generate_alerts(features_with_meta_df, threshold=0.5):
    df = features_with_meta_df[features_with_meta_df["score"] >= threshold]
    if df.empty:
        return pd.DataFrame(columns=["subnet", "pps", "score", "count", "color"])

    df["subnet"] = df["src_ip"].astype(str).apply(lambda x: ".".join(x.split(".")[:3]) + ".0/24")

    alerts = df.groupby("subnet").agg(
        pps=("pps", "mean"),
        score=("score", "mean"),
        count=("src_ip", "count")
    ).round(4).reset_index()

    alerts["color"] = alerts["score"].apply(lambda s: "red" if s >= 0.8 else ("yellow" if s >= 0.4 else "green"))
    return alerts.sort_values("score", ascending=False)

# ---------------------- MAIN ENGINE ----------------------
def realtime_engine():
    print("[ENGINE] Запуск real-time NDR-системы...")
    start_live_capture(config["live"]["interface"])

    buffer = []
    last_process = time.time()

    while True:
        # Сбор пакетов
        try:
            pkt = packet_queue.get(timeout=0.2)
            buffer.append(pkt)
        except:
            pass

        if time.time() - last_process < config["live"]["interval"]:
            time.sleep(0.01)
            continue
        if not buffer:
            last_process = time.time()
            continue

        print(f"[ENGINE] Обработка {len(buffer)} пакетов...")

        packets_df = pd.DataFrame(buffer)
        buffer.clear()

        # 1. Агрегация потоков
        flows = aggregate_flows(packets_df)
        if flows.empty:
            continue

        # Сохраняем IP до удаления
        flows_with_ip = flows[["src_ip", "dst_ip"]].copy()

        # 2. Извлечение признаков (они возвращают только числовые + label)
        features_numeric = extract_features(flows)

        # 3. Гарантируем все 45 признаков
        for col in FEATURES:
            if col not in features_numeric.columns:
                features_numeric[col] = 0.0
        X_numeric = features_numeric[FEATURES].copy()  # ← ТОЛЬКО ЧИСЛА!

        # 4. Считаем score
        scores = get_hybrid_score(X_numeric)

        # 5. Собираем финальный датафрейм с метаданными
        result_df = features_numeric.copy()
        result_df["score"] = scores
        result_df["src_ip"] = flows_with_ip["src_ip"].values
        result_df["dst_ip"] = flows_with_ip["dst_ip"].values

        # 6. Алерты
        threshold = config.get("ml", {}).get("threshold_yellow", 0.5)
        alerts_df = generate_alerts(result_df, threshold)

        # 7. Сохранение
        result_df.to_csv(OUTPUT_FLOWS, index=False)
        alerts_df.to_json(OUTPUT_ALERTS, orient="records", force_ascii=False)

        print(f"[ENGINE] Готово: {len(result_df)} потоков → {len(alerts_df)} алертов")

        last_process = time.time()


if __name__ == "__main__":
    try:
        realtime_engine()
    except KeyboardInterrupt:
        print("\n[ENGINE] Остановлено пользователем")