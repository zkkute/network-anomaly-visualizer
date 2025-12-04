import streamlit as st
import pandas as pd
import time
import altair as alt
from pathlib import Path

FLOWS_FILE = Path("data/live/flows_live.csv")
ALERTS_FILE = Path("data/live/alerts.json")

st.set_page_config(layout="wide")
st.title("🔴 Real-time Network Monitoring")

def load_data():
    flows = pd.read_csv(FLOWS_FILE) if FLOWS_FILE.exists() else pd.DataFrame()
    alerts = pd.read_json(ALERTS_FILE) if ALERTS_FILE.exists() else pd.DataFrame()
    return flows, alerts


placeholder = st.empty()

while True:
    flows, alerts = load_data()

    with placeholder.container():
        col1, col2 = st.columns(2)

        # Heatmap
        if not alerts.empty:
            chart = alt.Chart(alerts).mark_rect().encode(
                x="subnet:O",
                y="subnet:O",
                color="color:N",
                tooltip=["subnet", "score", "pps"]
            )
            col1.subheader("Heatmap")
            col1.altair_chart(chart, use_container_width=True)

        # Latest flows
        if not flows.empty:
            st.subheader("Последние 200 flow-ов:")
            st.dataframe(flows.tail(200))

    time.sleep(2)  # обновление
