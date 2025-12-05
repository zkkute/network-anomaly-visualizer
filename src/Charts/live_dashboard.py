import json
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Paths to data
# ----------------------------
FLOWS_CSV = r"C:/Users/Zlata/Desktop/network-anomaly-visualizer/data/live/flows_live.csv"
ALERTS_JSON = r"C:/Users/Zlata/Desktop/network-anomaly-visualizer/data/live/alerts.json"

# ----------------------------
# Dash App Initialization
# ----------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"backgroundColor": "#0E1117", "color": "white", "padding": "15px"},
    children=[
        html.H1("Network Monitoring Dashboard (Real-Time)", style={"textAlign": "center"}),

        dcc.Interval(id="update-timer", interval=2000, n_intervals=0),

        dcc.Graph(id="live-dashboard-graph", style={"height": "94vh"})
    ]
)

# ----------------------------
# Callback to update dashboard
# ----------------------------
@app.callback(
    Output("live-dashboard-graph", "figure"),
    Input("update-timer", "n_intervals")
)
def update_dashboard(_):

    # Load flows data
    try:
        df = pd.read_csv(FLOWS_CSV)
    except Exception:
        df = pd.DataFrame()

    # Load alerts
    try:
        with open(ALERTS_JSON, "r") as f:
            alerts = json.load(f)
    except:
        alerts = []

    # Replace missing score column if needed
    if "score" not in df.columns:
        df["score"] = 0

    # Sort by index (time)
    if not df.empty:
        df["index"] = df.index

    # Prepare basic aggregations
    top_src_ip = (
        df.groupby("src_ip")["score"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        if not df.empty else pd.Series()
    )

    top_pps = (
        df.groupby("src_ip")["pps"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        if not df.empty else pd.Series()
    )

    # Build dashboard layout
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "table"}, {"type": "indicator"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            "Rolling Avg & Max Score (by Flow Index)",
            "Top-10 Source IPs by Score",
            "Latest Alerts",
            "Mean Score (Gauge)",
            "Score Histogram",
            "Top-10 PPS by IP"
        ],
    )

    # 1. Rolling score over time
    if not df.empty:
        rolling_avg = df["score"].rolling(20).mean()
        rolling_max = df["score"].rolling(20).max()

        fig.add_trace(go.Scatter(
            x=df["index"], y=rolling_avg, mode="lines", name="Rolling Avg Score"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["index"], y=rolling_max, mode="lines", name="Rolling Max Score"
        ), row=1, col=1)

    # 2. Top anomalous source IPs
    fig.add_trace(go.Bar(
        x=top_src_ip.index, y=top_src_ip.values
    ), row=1, col=2)

    # 3. Alerts table
    fig.add_trace(
        go.Table(
            header=dict(values=["Timestamp", "IP", "Score"], fill_color="#202331"),
            cells=dict(values=[
                [a.get("timestamp", "") for a in alerts],
                [a.get("ip", "") for a in alerts],
                [a.get("score", "") for a in alerts],
            ])
        ),
        row=2, col=1
    )

    # 4. Score gauge
    mean_score = df["score"].mean() if not df.empty else 0
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=mean_score,
            gauge={"axis": {"range": [0, 10]}},
            title={"text": "Mean Score"},
        ),
        row=2, col=2
    )

    # 5. Score histogram
    if not df.empty:
        fig.add_trace(go.Histogram(
            x=df["score"], nbinsx=30
        ), row=3, col=1)

    # 6. Top PPS
    fig.add_trace(go.Bar(
        x=top_pps.index, y=top_pps.values
    ), row=3, col=2)

    fig.update_layout(
        height=900,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig


# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    print("Starting Dash server on http://127.0.0.1:8050 ...")
    app.run(debug=False)
