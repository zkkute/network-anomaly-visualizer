#src/visualization/app_live.py
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from streamlit_autorefresh import st_autorefresh   # <-- Библиотека автообновления

FLOWS_FILE = Path("data/live/flows_live.csv")
ALERTS_FILE = Path("data/live/alerts.json")

st.set_page_config(layout="wide")
st.title("🔴 Real-time Network Monitoring Dashboard")

# Автообновление каждые 2 секунды
st_autorefresh(interval=2000, key="live_refresh")

def load_data():
    flows = pd.read_csv(FLOWS_FILE) if FLOWS_FILE.exists() else pd.DataFrame()
    alerts = pd.read_json(ALERTS_FILE) if ALERTS_FILE.exists() else pd.DataFrame()
    return flows, alerts

flows, alerts = load_data()

col1, col2 = st.columns([2, 1])

# ---------------- HEATMAP ----------------
col1.subheader("Heatmap of Subnets")

if not alerts.empty:
    chart = alt.Chart(alerts).mark_rect().encode(
        x=alt.X("subnet:N", title="Subnet"),
        y=alt.value(0),
        color=alt.Color("score:Q",
                        scale=alt.Scale(domain=[0.0, 1.0], scheme="redyellowgreen"),
                        title="Anomaly Score"),
        tooltip=["subnet", "score", "pps", "count"]
    ).properties(
        height=80,
        width="container"
    )

    col1.altair_chart(chart, use_container_width=True)
else:
    col1.info("Пока нет аномальных подсетей.")


# ---------------- FLOW TABLE ----------------
col2.subheader("Последние 200 Flow-ов")

if not flows.empty:
    col2.dataframe(flows.tail(200))
else:
    col2.info("Данные еще не поступили.")
