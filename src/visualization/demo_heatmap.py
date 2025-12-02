# src/visualization/demo_heatamap.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import joblib
import xgboost as xgb
import hashlib
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.options.mode.chained_assignment = None  # Отключает SettingWithCopyWarning

# И ещё один важный — убирает предупреждения от XGBoost
import os
os.environ["XGBoost_WARNING"] = "0"

st.set_page_config(page_title="Network Anomaly Visualizer", layout="wide")
st.title("Система обнаружения сетевых аномалий")
st.markdown("### Дипломный проект — Злата")

# ------------------- ИНИЦИАЛИЗАЦИЯ СЕССИИ -------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "grid" not in st.session_state:
    st.session_state.grid = np.zeros((16, 16))
if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0
if "total_anomalies" not in st.session_state:
    st.session_state.total_anomalies = 0

# ------------------- ЗАГРУЗКА -------------------
@st.cache_resource
def load_models_and_data():
    paths = {
        "scaler": Path("models/scaler.pkl"),
        "features": Path("models/feature_list.pkl"),
        "xgb": Path("models/xgboost_model.json"),
        "iso": Path("models/isolation_forest.pkl"),
        "data": Path("data/processed/my_traffic_features.csv")
    }

    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        st.error(f"Не найдены: {', '.join(missing)}")
        st.code("python scripts/preprocess_universal.py\npython scripts/train_models.py")
        st.stop()

    scaler = joblib.load(paths["scaler"])
    FEATURES = joblib.load(paths["features"])
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(paths["xgb"])
    iso_model = joblib.load(paths["iso"])
    df = pd.read_csv(paths["data"])
    if 'label' not in df.columns:
        df['label'] = 'normal'

    return scaler, FEATURES, xgb_model, iso_model, df

scaler, FEATURES, xgb_model, iso_model, df_demo = load_models_and_data()

st.success("Всё загружено!")
st.write(f"Потоков: **{len(df_demo):,}** | Фич: **{len(FEATURES)}**")

# ------------------- SIDEBAR -------------------
st.sidebar.header("Настройки")
batch_size = st.sidebar.slider("Батч", 100, 3000, 800, 100)
speed = st.sidebar.slider("Скорость (сек)", 0.01, 0.5, 0.08, 0.01)
threshold = st.sidebar.slider("Порог аномалии", 0.5, 0.95, 0.80, 0.05)

# ------------------- КНОПКИ УПРАВЛЕНИЯ -------------------
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("ЗАПУСТИТЬ", type="primary", use_container_width=True):
    st.session_state.running = True
    st.session_state.paused = False
    st.session_state.grid = np.zeros((16, 16))
    st.session_state.total_processed = 0
    st.session_state.total_anomalies = 0
    st.rerun()

if col2.button("ПАУЗА" if not st.session_state.paused else "ПРОДОЛЖИТЬ",
               type="secondary" if not st.session_state.paused else "primary",
               disabled=not st.session_state.running,
               use_container_width=True):
    st.session_state.paused = not st.session_state.paused
    st.rerun()

if col3.button("СТОП", type="secondary", disabled=not st.session_state.running, use_container_width=True):
    st.session_state.running = False
    st.session_state.paused = False
    st.rerun()

# ------------------- ДЕМО -------------------
if st.session_state.running:
    st.markdown("### Тепловая карта аномалий")

    # Плейсхолдеры — остаются на месте!
    chart = st.empty()
    stats = st.empty()
    progress_bar = st.progress(0)

    start_time = time.time()
    idx = st.session_state.get("current_idx", 0)

    while idx < len(df_demo) and st.session_state.running:
        # Пауза — просто ждём
        while st.session_state.paused and st.session_state.running:
            time.sleep(0.1)
            st.rerun()

        end_idx = min(idx + batch_size, len(df_demo))
        batch = df_demo.iloc[idx:end_idx].copy()

        # Подготовка данных
        X_input = pd.DataFrame(0.0, columns=FEATURES, index=batch.index)
        for col in batch.columns:
            if col in FEATURES:
                X_input[col] = batch[col]
        X_scaled = scaler.transform(X_input)

        # Предсказания
        xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
        iso_raw = -iso_model.decision_function(X_scaled)
        iso_score = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-12)
        final_score = np.maximum(xgb_proba, iso_score)

        # Статистика
        batch_anoms = (final_score >= threshold).sum()
        st.session_state.total_processed += len(batch)
        st.session_state.total_anomalies += batch_anoms

        # Обновляем сетку
        for i, score in enumerate(final_score):
            h = hashlib.md5(str(idx + i).encode()).hexdigest()
            x = int(h[:8], 16) % 16
            y = int(h[8:16], 16) % 16
            st.session_state.grid[x, y] = max(st.session_state.grid[x, y], score)

        st.session_state.grid *= 0.96

        # График
        fig = go.Figure(data=go.Heatmap(
            z=st.session_state.grid.T,
            colorscale=[[0, "#00ff00"], [0.6, "yellow"], [1, "#ff0000"]],
            zmin=0, zmax=1, showscale=False
        ))
        fig.update_layout(width=900, height=650, plot_bgcolor='black', paper_bgcolor='black', margin=dict(t=40))
        chart.plotly_chart(fig, use_container_width=True)

        # Статистика
        stats.markdown(f"""
        ### Статистика
        | Показатель                | Значение                      |
        |--------------------------|-------------------------------|
        | Обработано потоков       | **{st.session_state.total_processed:,}** |
        | Всего аномалий           | **{st.session_state.total_anomalies:,}** |
        | Аномалий в этом батче    | **{batch_anoms}**             |
        | Порог                    | **{threshold:.2f}**           |
        | Макс. score              | **{final_score.max():.4f}**   |
        | Прогресс                 | **{(st.session_state.total_processed / len(df_demo) * 100):.1f}%** |
        """, unsafe_allow_html=True)

        progress_bar.progress(st.session_state.total_processed / len(df_demo))

        idx = end_idx
        st.session_state.current_idx = idx

        time.sleep(speed)

    # Завершение
    if st.session_state.running:
        st.session_state.running = False
        st.success(f"ДЕМО ЗАВЕРШЕНО! Обнаружено аномалий: **{st.session_state.total_anomalies:,}**")
        st.balloons()
    st.rerun()

# ------------------- ИНФО -------------------
with st.expander("О датасете"):
    st.write(f"Всего потоков: **{len(df_demo):,}**")
    st.write("Метки:", dict(df_demo['label'].value_counts()))
    if 'attack' in df_demo['label'].str.lower().values:
        st.warning("Есть аномалии — будет красное!")
    else:
        st.success("Только нормальный трафик — будет зелёное море")