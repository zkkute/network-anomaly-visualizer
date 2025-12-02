# src/visualization/demo_heatmap.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import joblib
import xgboost as xgb
import hashlib
from pathlib import Path
import threading
from queue import Empty

# === Импорты из твоего проекта ===
from src.ingestion.live_capture import packet_queue, start_live_capture
from src.preprocessing.feature_extractor import extract_features

# === Настройка страницы ===
st.set_page_config(
    page_title="Live Network Anomaly Detector",
    page_icon="Warning",
    layout="wide"
)

st.title("Live + Оффлайн: Система обнаружения аномалий в реальном времени")
st.markdown("### Гибридная NIDS: XGBoost + Isolation Forest | Диплом 2025 — Злата")

# === Инициализация состояния ===
for key, default in [
    ("grid", np.zeros((16, 16))),
    ("total_packets", 0),
    ("total_anomalies", 0),
    ("live_running", False),
    ("current_idx", 0)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# === Загрузка моделей (с кэшированием) ===
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("models/scaler.pkl")
        FEATURES = joblib.load("models/feature_list.pkl")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("models/xgboost_model.json")
        iso_model = joblib.load("models/isolation_forest.pkl")
        return scaler, FEATURES, xgb_model, iso_model
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
        st.code("python scripts/train_models.py")
        st.stop()

scaler, FEATURES, xgb_model, iso_model = load_models()

# === Sidebar ===
st.sidebar.header("Управление системой")
mode = st.sidebar.radio("Режим работы", ["Live захват (реальное время)", "Оффлайн демо (CSV)"], index=0)

threshold = st.sidebar.slider("Порог обнаружения", 0.5, 0.95, 0.80, 0.01)
fade_rate = st.sidebar.slider("Затухание пятна", 0.90, 0.99, 0.96, 0.01)
batch_interval = st.sidebar.slider("Интервал обработки (сек)", 1, 10, 3)

# === LIVE РЕЖИМ ===
if mode == "Live захват (реальное время)":
    st.sidebar.info("Запусти атаку (nmap, hping3) — увидишь красное пятно!")

    interface = st.sidebar.text_input("Интерфейс (оставь пустым = авто)", "")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("ЗАПУСТИТЬ LIVE", type="primary", use_container_width=True):
        st.session_state.live_running = True
        st.session_state.grid = np.zeros((16, 16))
        st.session_state.total_packets = 0
        st.session_state.total_anomalies = 0
        threading.Thread(target=start_live_capture, args=(interface or None,), daemon=True).start()
        st.rerun()

    if col2.button("СТОП", type="secondary", use_container_width=True):
        st.session_state.live_running = False
        st.rerun()

    if st.session_state.live_running:
        chart = st.empty()
        stats = st.empty()

        packets_buffer = []
        last_update = time.time()

        while st.session_state.live_running:
            # Сбор пакетов за интервал
            while time.time() - last_update < batch_interval:
                try:
                    pkt = packet_queue.get_nowait()
                    packets_buffer.append(pkt)
                    st.session_state.total_packets += 1
                except Empty:
                    time.sleep(0.01)
                except Exception:
                    time.sleep(0.01)

            if packets_buffer:
                df_batch = pd.DataFrame(packets_buffer)
                packets_buffer.clear()

                try:
                    df_flows = extract_features(df_batch)

                    if len(df_flows) > 0:
                        # Подгоняем под ожидаемые фичи
                        X = pd.DataFrame(0.0, columns=FEATURES, index=df_flows.index)
                        for col in FEATURES:
                            if col in df_flows.columns:
                                X[col] = df_flows[col]

                        X_scaled = scaler.transform(X)

                        # Предсказания
                        xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
                        iso_raw = -iso_model.decision_function(X_scaled)
                        iso_score = (iso_raw - iso_raw.min()) / (iso_raw.ptp() + 1e-12)
                        final_score = np.maximum(xgb_proba, iso_score)

                        # Обновление тепловой карты
                        for score in final_score:
                            if score >= threshold:
                                h = hashlib.md5(str(time.time_ns()).encode()).hexdigest()
                                x = int(h[:8], 16) % 16
                                y = int(h[8:16], 16) % 16
                                st.session_state.grid[x, y] = max(st.session_state.grid[x, y], score)

                        st.session_state.grid *= fade_rate
                        st.session_state.total_anomalies += int((final_score >= threshold).sum())

                except Exception as e:
                    pass  # Пропускаем ошибки обработки батча

            last_update = time.time()

            # === Визуализация ===
            fig = go.Figure(data=go.Heatmap(
                z=st.session_state.grid.T,
                colorscale=[[0, "#00ff00"], [0.6, "yellow"], [1, "#ff0000"]],
                zmin=0, zmax=1,
                showscale=False
            ))
            fig.update_layout(
                width=1000, height=700,
                plot_bgcolor='black',
                paper_bgcolor='black',
                title=f"LIVE: {st.session_state.total_packets:,} пакетов | {st.session_state.total_anomalies:,} аномалий"
            )
            chart.plotly_chart(fig, use_container_width=True)

            stats.markdown(f"""
            **Live статистика**  
            Пакетов обработано: **{st.session_state.total_packets:,}**  
            Аномалий обнаружено: **{st.session_state.total_anomalies:,}**  
            Порог: **{threshold:.2f}** | Затухание: **{fade_rate:.3f}**  
            Интервал: **{batch_interval} сек**  
            """)

            time.sleep(0.1)
            st.rerun()

        st.success("Live-режим остановлен")
        st.balloons()

    else:
        st.info("Нажми **ЗАПУСТИТЬ LIVE** и запусти атаку (nmap, hping3 и т.д.)")

# === ОФФЛАЙН ДЕМО (твой старый код, но компактный) ===
else:
    st.info("Оффлайн демо на датасете")

    @st.cache_data
    def load_demo_data():
        path = Path("data/processed/my_traffic_features.csv")
        if path.exists():
            return pd.read_csv(path)
        else:
            st.error("Нет файла data/processed/my_traffic_features.csv")
            st.stop()

    df_demo = load_demo_data()

    if st.button("Запустить оффлайн демо", type="primary"):
        st.session_state.grid = np.zeros((16, 16))
        st.session_state.total_processed = 0
        st.session_state.total_anomalies = 0
        st.session_state.current_idx = 0
        st.rerun()

    if st.session_state.get("current_idx", 0) < len(df_demo):
        chart = st.empty()
        stats = st.empty()
        progress = st.progress(0)

        batch_size = 800
        idx = st.session_state.current_idx

        while idx < len(df_demo):
            end_idx = min(idx + batch_size, len(df_demo))
            batch = df_demo.iloc[idx:end_idx]

            X = pd.DataFrame(0.0, columns=FEATURES, index=batch.index)
            for col in FEATURES:
                if col in batch.columns:
                    X[col] = batch[col]
            X_scaled = scaler.transform(X)

            xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
            iso_raw = -iso_model.decision_function(X_scaled)
            iso_score = (iso_raw - iso_raw.min()) / (iso_raw.ptp() + 1e-12)
            final_score = np.maximum(xgb_proba, iso_score)

            for i, score in enumerate(final_score):
                if score >= threshold:
                    h = hashlib.md5(str(idx + i).encode()).hexdigest()
                    x = int(h[:8], 16) % 16
                    y = int(h[8:16], 16) % 16
                    st.session_state.grid[x, y] = max(st.session_state.grid[x, y], score)

            st.session_state.grid *= fade_rate
            st.session_state.total_anomalies += int((final_score >= threshold).sum())
            st.session_state.total_processed += len(batch)
            st.session_state.current_idx = end_idx

            fig = go.Figure(data=go.Heatmap(z=st.session_state.grid.T,
                                            colorscale=[[0, "#00ff00"], [0.6, "yellow"], [1, "#ff0000"]],
                                            zmin=0, zmax=1, showscale=False))
            fig.update_layout(width=1000, height=700, plot_bgcolor='black', paper_bgcolor='black')
            chart.plotly_chart(fig, use_container_width=True)

            stats.markdown(f"Обработано: {st.session_state.total_processed:,} | Аномалий: {st.session_state.total_anomalies:,}")
            progress.progress(st.session_state.total_processed / len(df_demo))

            time.sleep(0.08)
            st.rerun()

        st.success("Демо завершено!")
        st.balloons()

st.caption("Гибридная система обнаружения сетевых аномалий | Злата, 2025")