# src/Charts/live_dashboard.py
# Живая панель NDR — работает сразу, без ошибок

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

FLOWS_FILE = Path("data/live/flows_live.csv")
ALERTS_FILE = Path("data/live/alerts.json")

print("NDR Live Dashboard запущен")
print("Ожидание данных от realtime_engine...")

history = []  # для графиков по времени

while True:
    try:
        if not FLOWS_FILE.exists():
            print("flows_live.csv не найден — запусти realtime_engine")
            time.sleep(3)
            continue

        df = pd.read_csv(FLOWS_FILE)

        alerts = pd.DataFrame()
        if ALERTS_FILE.exists():
            try:
                alerts = pd.read_json(ALERTS_FILE)
            except:
                pass

        current_time = pd.Timestamp.now().strftime("%H:%M:%S")
        total_flows = len(df)
        avg_score = df['score'].mean() if 'score' in df.columns and not df.empty else 0
        max_score = df['score'].max() if 'score' in df.columns and not df.empty else 0

        # Топ-10 IP по аномальности
        top_ips = (df.nlargest(10, 'score')[['src_ip', 'score', 'pps']]
                   if 'score' in df.columns and not df.empty else pd.DataFrame())

        # История
        history.append({'time': current_time, 'avg': avg_score, 'max': max_score, 'alerts': len(alerts)})
        if len(history) > 60:
            history = history[-60:]
        hist = pd.DataFrame(history)

        # === Дашборд ===
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Динамика аномальности", "Топ-10 подозрительных IP",
                "Алерты по подсетям", "Средний anomaly score",
                "Распределение score", "PPS у самых аномальных"
            ),
            specs=[
                [{}, {"type": "bar"}],
                [{"type": "table"}, {"type": "indicator"}],
                [{"type": "histogram"}, {"type": "bar"}]
            ]
        )

        # 1. Динамика
        if len(hist) > 1:
            fig.add_trace(go.Scatter(x=hist['time'], y=hist['avg'],
                                     mode='lines+markers', name='Средний score',
                                     line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist['time'], y=hist['max'],
                                     mode='lines+markers', name='Максимальный score',
                                     line=dict(color='red')), row=1, col=1)

        # 2. Топ IP
        if not top_ips.empty:
            fig.add_trace(go.Bar(
                x=top_ips['src_ip'].astype(str),
                y=top_ips['score'],
                text=top_ips['score'].round(3),
                textposition='outside',
                marker_color='crimson',
                name='Anomaly Score'
            ), row=1, col=2)

        # 3. Таблица алертов
        if not alerts.empty:
            top_alerts = alerts.head(10)
            fig.add_trace(go.Table(
                header=dict(values=["Подсеть", "Score", "PPS", "IP-ов"], font=dict(size=12), fill_color='grey'),
                cells=dict(
                    values=[
                        top_alerts['subnet'],
                        top_alerts['score'].round(3),
                        top_alerts['pps'].round(1),
                        top_alerts['count']
                    ],
                    fill_color=[
                        ['lightgreen' if c == 'green' else 'yellow' if c == 'yellow' else 'lightcoral'
                         for c in top_alerts['color']]
                    ],
                    font=dict(size=11)
                )
            ), row=2, col=1)

        # 4. Индикатор среднего score
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=avg_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Средний score"},
            delta={'reference': 0.3},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.4], 'color': "lightgreen"},
                    {'range': [0.4, 0.8], 'color': "yellow"},
                    {'range': [0.8, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ), row=2, col=2)

        # 5. Гистограмма
        if 'score' in df.columns and not df.empty():
            fig.add_trace(go.Histogram(
                x=df['score'],
                nbinsx=40,
                name="Score",
                marker_color='cyan'
            ), row=3, col=1)

        # 6. PPS топ IP
        if not top_ips.empty:
            fig.add_trace(go.Bar(
                x=top_ips['src_ip'].astype(str),
                y=top_ips['pps'],
                name="PPS",
                marker_color='limegreen'
            ), row=3, col=2)

        fig.update_layout(
            height=1100,
            title_text=f"NDR LIVE DASHBOARD | {current_time} | Потоков: {total_flows} | Алертов: {len(alerts)}",
            template="plotly_dark",
            showlegend=False
        )

        fig.show()

        print(f"Обновлено {current_time} | Score: {avg_score:.3f} | Макс: {max_score:.3f} | Алертов: {len(alerts)}")

    except Exception as e:
        print(f"Ошибка: {e}")

    time.sleep(2)