# src/visualization/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import joblib
import xgboost as xgb
import hashlib

st.set_page_config(page_title="Network Anomaly Visualizer", layout="wide")
st.title("Система обнаружения сетевых аномалий")
st.markdown("### Дипломный проект — Злата")


# ------------------- ЗАГРУЗКА -------------------
@st.cache_resource
def load_everything():
    df = pd.read_csv("data/processed/friday_features.csv")

    # Важно: сохраняем имена колонок фич
    feature_columns = [col for col in df.columns if col != 'label']

    scaler = joblib.load("models/scaler.pkl")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_model.json")
    iso_model = joblib.load("models/isolation_forest.pkl")

    return df, scaler, xgb_model, iso_model, feature_columns


df_full, scaler, xgb_model, iso_model, FEATURES = load_everything()

# ------------------- SIDEBAR -------------------
st.sidebar.header("Настройки демо")
batch_size = st.sidebar.slider("Размер батча", 500, 3000, 1200, 100)
speed = st.sidebar.slider("Скорость (сек)", 0.01, 0.20, 0.05, 0.01)

# ------------------- ДЕМО -------------------
if st.sidebar.button("ЗАПУСТИТЬ ДЕМО — Friday с DDoS и PortScan", type="primary"):
    st.success("Демо запущено! Ждите красное пятно при DDoS")

    grid = np.zeros((16, 16))
    progress = st.progress(0)
    status = st.empty()
    chart_placeholder = st.empty()

    total = len(df_full)

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = df_full.iloc[start_idx:end_idx]

        # Важно: оставляем DataFrame с теми же именами колонок!
        X_scaled = pd.DataFrame(
            scaler.transform(batch.drop('label', axis=1)),
            columns=FEATURES,
            index=batch.index
        )

        # Теперь никаких warning!
        xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
        iso_score_raw = -iso_model.decision_function(X_scaled)
        iso_score = (iso_score_raw - iso_score_raw.min()) / (iso_score_raw.ptp() + 1e-8)
        final_score = np.maximum(xgb_proba, iso_score)

        # Раскидываем по сетке 16×16
        for i, score in enumerate(final_score):
            idx = start_idx + i
            h = hashlib.md5(str(idx).encode()).hexdigest()
            x = int(h[:8], 16) % 16
            y = int(h[8:16], 16) % 16
            grid[x, y] = max(grid[x, y], score)

        # Затухание
        grid *= 0.94

        # График
        fig = go.Figure(data=go.Heatmap(
            z=grid,
            colorscale=[[0, "green"], [0.6, "yellow"], [1, "red"]],
            zmin=0, zmax=1,
            showscale=False
        ))
        fig.update_layout(
            title=f"Пятница | Обработано {end_idx:,}/{total:,} | Атак в окне: {(final_score > 0.8).sum()}",
            width=1000, height=700
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Метрики
        progress.progress(end_idx / total)
        status.markdown(f"""
        **Прогресс:** {end_idx / total * 100:,.1f}%  
        **Max score:** {final_score.max():.4f}  
        **Обнаружено атак:** {(final_score > 0.8).sum()}
        """)

        time.sleep(speed)

    st.success("ДЕМО ЗАВЕРШЕНО! Красное пятно = DDoS атака")
    st.balloons()
    st.markdown("### Готово!")