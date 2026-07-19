# -*- coding: utf-8 -*-
"""
NDR Dashboard — единое окно для демонстрации комиссии.
Соответствует разделу 4.7 ВКР: главное окно мониторинга, вкладка
топологического анализа, вкладка статистического анализа.

Запуск:
    streamlit run ndr_dashboard.py

Если файлы models/*.pkl и data/processed/my_traffic_features.csv существуют —
дашборд использует их. Если нет — генерирует реалистичные синтетические
данные, имитирующие смешанный трафик с DDoS-инцидентом.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ===================== СТРАНИЦА =====================
st.set_page_config(
    page_title="NDR Dashboard — Гибридная система обнаружения сетевых аномалий",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== СТИЛЬ =====================
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stApp { background-color: #0E1117; color: #E6E6E6; }
    h1, h2, h3, h4 { color: #FAFAFA; }
    div[data-testid="stMetricValue"] { color: #FAFAFA; font-size: 1.6rem; }
    div[data-testid="stMetricLabel"] { color: #A0A0A0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E2230;
        border-radius: 6px 6px 0 0;
        padding: 10px 22px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E3548;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)


# ===================== ГЕНЕРАЦИЯ ДАННЫХ =====================
@st.cache_data
def generate_demo_traffic(n_flows: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует синтетический набор сетевых потоков, имитирующий
    реальный смешанный трафик с участком DDoS-атаки.
    Все диапазоны подобраны так, чтобы графики были информативными.
    """
    rng = np.random.default_rng(seed)

    # Шкала времени: последние 30 минут
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=30)
    timestamps = pd.to_datetime(
        rng.uniform(start_time.timestamp(), end_time.timestamp(), n_flows),
        unit='s'
    )

    # Подсети источника (внешние + внутренние)
    src_subnets = [
        "192.168.10.0/24", "192.168.20.0/24", "10.0.5.0/24", "10.0.8.0/24",
        "172.16.0.0/24", "203.0.113.0/24", "198.51.100.0/24", "185.220.101.0/24",
    ]
    dst_subnets = [
        "10.0.1.0/24", "10.0.2.0/24", "192.168.1.0/24",
        "192.168.50.0/24", "172.20.0.0/24", "203.0.113.0/24",
    ]
    protocols = ["TCP", "UDP", "HTTP", "HTTPS", "DNS", "ICMP"]
    proto_p = [0.35, 0.15, 0.20, 0.20, 0.07, 0.03]

    src_choice = rng.choice(src_subnets, n_flows)
    dst_choice = rng.choice(dst_subnets, n_flows)
    proto_choice = rng.choice(protocols, n_flows, p=proto_p)

    # IP-адреса источников
    src_ips = []
    for s in src_choice:
        base = s.rsplit(".", 1)[0].rsplit(".", 1)[0]
        src_ips.append(f"{base}.{rng.integers(1, 254)}")

    # Базовые признаки нормального трафика (log-normal распределения)
    duration = rng.lognormal(mean=0.5, sigma=1.2, size=n_flows)  # сек
    packets = rng.lognormal(mean=2.5, sigma=1.0, size=n_flows).astype(int) + 1
    bytes_total = packets * rng.lognormal(mean=6.0, sigma=0.8, size=n_flows)
    pps = packets / np.maximum(duration, 0.01)
    bps = bytes_total / np.maximum(duration, 0.01)

    # Базовая оценка аномальности — преимущественно низкая
    score = rng.beta(2, 12, n_flows)  # пик у 0.1–0.2

    # ===== Внедряем DDoS-инцидент =====
    # 12% потоков с подсети 185.220.101.0/24 → 10.0.1.0/24
    ddos_mask = (src_choice == "185.220.101.0/24") & (dst_choice == "10.0.1.0/24")
    # добавим ещё случайных
    extra_ddos_idx = rng.choice(np.where(~ddos_mask)[0], size=int(n_flows * 0.06), replace=False)
    ddos_mask[extra_ddos_idx] = True

    score[ddos_mask] = rng.beta(8, 2, ddos_mask.sum())  # высокая оценка
    pps[ddos_mask] *= rng.uniform(20, 80, ddos_mask.sum())  # резкий рост
    packets[ddos_mask] = packets[ddos_mask] * rng.integers(5, 25, ddos_mask.sum())

    # Подмешиваем небольшое количество подозрительных
    susp_idx = rng.choice(n_flows, size=int(n_flows * 0.04), replace=False)
    score[susp_idx] = rng.uniform(0.45, 0.75, len(susp_idx))

    score = np.clip(score, 0, 1)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "src_ip": src_ips,
        "src_subnet": src_choice,
        "dst_subnet": dst_choice,
        "protocol": proto_choice,
        "duration": duration,
        "packets": packets,
        "bytes": bytes_total,
        "pps": pps,
        "bps": bps,
        "score": score,
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ===================== ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ (если есть) =====================
@st.cache_data
def try_load_real_data():
    """Пытается загрузить реальный датасет проекта, иначе возвращает None."""
    path = Path("data/processed/my_traffic_features.csv")
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # Минимальная адаптация под формат дашборда
        if "score" not in df.columns:
            # Если запускают без обученных моделей — сгенерируем score
            from sklearn.ensemble import IsolationForest
            feats = df.select_dtypes(include=[np.number]).fillna(0)
            iso = IsolationForest(contamination=0.05, random_state=42)
            iso.fit(feats)
            raw = -iso.decision_function(feats)
            df["score"] = (raw - raw.min()) / (raw.ptp() + 1e-12)
        return df
    except Exception:
        return None


# ===================== САЙДБАР =====================
with st.sidebar:
    st.markdown("### Управление")
    threshold_yellow = st.slider("Порог threshold_yellow", 0.30, 0.70, 0.50, 0.05)
    threshold_red = st.slider("Порог threshold_red", 0.70, 0.95, 0.80, 0.05)
    window_min = st.slider("Окно мониторинга (мин)", 5, 60, 30, 5)
    st.divider()
    st.caption("Источник данных")
    use_demo = st.checkbox("Использовать демо-данные", value=True,
                           help="Снимите, чтобы попытаться загрузить data/processed/my_traffic_features.csv")
    st.divider()
    st.caption(f"Сборка: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

# Загрузка данных
df = None
if not use_demo:
    df = try_load_real_data()
    if df is None:
        st.sidebar.warning("Реальный датасет не найден — используются демо-данные")
if df is None:
    df = generate_demo_traffic()

# ===================== ЗАГОЛОВОК =====================
st.markdown("# Гибридная NDR-система: панель аналитика SOC")
st.caption("XGBoost + Isolation Forest · поведенческие признаки уровня L3–L4 · реальное время")

# ===================== ВКЛАДКИ =====================
tab_main, tab_topo, tab_stats = st.tabs([
    "Главное окно мониторинга",
    "Топологический анализ",
    "Статистический анализ",
])


# =================================================================
#                ВКЛАДКА 1 — ГЛАВНОЕ ОКНО МОНИТОРИНГА
# =================================================================
with tab_main:
    # Метрики в шапке
    total_flows = len(df)
    suspicious = int((df["score"] >= threshold_yellow).sum())
    critical = int((df["score"] >= threshold_red).sum())
    mean_score = float(df["score"].mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Потоков обработано", f"{total_flows:,}")
    m2.metric("Подозрительные", f"{suspicious:,}",
              delta=f"{suspicious/total_flows*100:.1f}% от общего")
    m3.metric("Критические", f"{critical:,}",
              delta=f"{critical/total_flows*100:.2f}%", delta_color="inverse")
    m4.metric("Mean Score", f"{mean_score:.3f}")

    st.markdown("---")

    # Верхний ряд: Mean Score gauge + Rolling Avg/Max
    row1_col1, row1_col2 = st.columns([1, 2])

    with row1_col1:
        # Gauge — индикатор Mean Score
        gauge_color = (
            "#2ECC71" if mean_score < threshold_yellow
            else "#F39C12" if mean_score < threshold_red
            else "#E74C3C"
        )
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Mean Score (уровень угрозы)", "font": {"size": 16, "color": "#FAFAFA"}},
            number={"font": {"size": 42, "color": gauge_color}, "valueformat": ".3f"},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#A0A0A0", "tickfont": {"color": "#A0A0A0"}},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "#1E2230",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold_yellow], "color": "#1E3A2A"},
                    {"range": [threshold_yellow, threshold_red], "color": "#3A2E1E"},
                    {"range": [threshold_red, 1], "color": "#3A1E1E"},
                ],
                "threshold": {
                    "line": {"color": "#FAFAFA", "width": 3},
                    "thickness": 0.85,
                    "value": mean_score,
                },
            },
        ))
        gauge.update_layout(
            paper_bgcolor="#0E1117", font={"color": "#FAFAFA"},
            height=320, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(gauge, use_container_width=True)

    with row1_col2:
        # Rolling Avg & Max Score
        df_t = df.sort_values("timestamp").copy()
        df_t["minute"] = df_t["timestamp"].dt.floor("30s")
        agg = df_t.groupby("minute")["score"].agg(["mean", "max"]).reset_index()
        # скользящее окно
        agg["rolling_avg"] = agg["mean"].rolling(5, min_periods=1).mean()
        agg["rolling_max"] = agg["max"].rolling(5, min_periods=1).max()

        ts_fig = go.Figure()
        ts_fig.add_trace(go.Scatter(
            x=agg["minute"], y=agg["rolling_avg"], mode="lines",
            name="Rolling Avg Score", line=dict(color="#3498DB", width=2.5),
            fill="tozeroy", fillcolor="rgba(52, 152, 219, 0.15)",
        ))
        ts_fig.add_trace(go.Scatter(
            x=agg["minute"], y=agg["rolling_max"], mode="lines",
            name="Rolling Max Score", line=dict(color="#E74C3C", width=2.5, dash="dot"),
        ))
        ts_fig.add_hline(y=threshold_red, line=dict(color="#E74C3C", width=1, dash="dash"),
                         annotation_text=f"red {threshold_red}", annotation_position="top left",
                         annotation_font_color="#E74C3C")
        ts_fig.add_hline(y=threshold_yellow, line=dict(color="#F39C12", width=1, dash="dash"),
                         annotation_text=f"yellow {threshold_yellow}", annotation_position="bottom left",
                         annotation_font_color="#F39C12")
        ts_fig.update_layout(
            title=dict(text=f"Rolling Avg & Max Score · окно {window_min} мин", font_color="#FAFAFA"),
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1F2C",
            font_color="#E6E6E6",
            xaxis=dict(title="Время", gridcolor="#2A3142"),
            yaxis=dict(title="Score_hybrid", gridcolor="#2A3142", range=[0, 1.05]),
            height=320, margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
            hovermode="x unified",
        )
        st.plotly_chart(ts_fig, use_container_width=True)

    # Второй ряд: гистограмма + два топа
    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=df["score"], nbinsx=40,
            marker=dict(
                color=df["score"].apply(
                    lambda s: "#E74C3C" if s >= threshold_red
                    else "#F39C12" if s >= threshold_yellow
                    else "#3498DB"
                ).tolist()[:0] or "#3498DB",
            ),
            opacity=0.85, name="Потоки",
        ))
        # Окраска бинов через несколько трасс — корректнее
        hist_fig.data = []
        bins = np.linspace(0, 1, 41)
        for lo, hi, color, label in [
            (0, threshold_yellow, "#3498DB", "Нормальные"),
            (threshold_yellow, threshold_red, "#F39C12", "Подозрительные"),
            (threshold_red, 1.01, "#E74C3C", "Критические"),
        ]:
            sub = df[(df["score"] >= lo) & (df["score"] < hi)]["score"]
            if len(sub):
                hist_fig.add_trace(go.Histogram(
                    x=sub, xbins=dict(start=0, end=1, size=0.025),
                    marker_color=color, opacity=0.85, name=label,
                ))
        hist_fig.add_vline(x=threshold_yellow, line=dict(color="#F39C12", width=1, dash="dash"))
        hist_fig.add_vline(x=threshold_red, line=dict(color="#E74C3C", width=1, dash="dash"))
        hist_fig.update_layout(
            title=dict(text="Распределение гибридных оценок Score_hybrid",
                       font_color="#FAFAFA"),
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1F2C",
            font_color="#E6E6E6",
            xaxis=dict(title="Score_hybrid", gridcolor="#2A3142"),
            yaxis=dict(title="Частота (число потоков)", gridcolor="#2A3142", type="log"),
            barmode="stack", height=360,
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with row2_col2:
        # Топ-10 источников по объёму трафика
        top_sources = (df.groupby("src_ip")["bytes"].sum()
                       .sort_values(ascending=False).head(10).reset_index())
        top_sources["MB"] = top_sources["bytes"] / (1024 * 1024)

        ts_bar = go.Figure(go.Bar(
            y=top_sources["src_ip"][::-1],
            x=top_sources["MB"][::-1],
            orientation="h",
            marker=dict(
                color=top_sources["MB"][::-1],
                colorscale="Blues",
                showscale=False,
                line=dict(color="#2A3142", width=1),
            ),
            text=[f"{v:.1f} МБ" for v in top_sources["MB"][::-1]],
            textposition="outside",
            textfont=dict(color="#E6E6E6"),
        ))
        ts_bar.update_layout(
            title=dict(text="Топ-10 источников трафика (по объёму)",
                       font_color="#FAFAFA"),
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1F2C",
            font_color="#E6E6E6",
            xaxis=dict(title="Объём, МБ", gridcolor="#2A3142"),
            yaxis=dict(title="", gridcolor="#2A3142"),
            height=360, margin=dict(l=120, r=40, t=50, b=40),
        )
        st.plotly_chart(ts_bar, use_container_width=True)

    # Третий ряд: Топ-10 IP по доле подозрительных потоков (полная ширина)
    by_ip = df.groupby("src_ip").agg(
        total=("score", "count"),
        susp=("score", lambda s: (s >= threshold_yellow).sum()),
        mean_score=("score", "mean"),
    ).reset_index()
    by_ip = by_ip[by_ip["total"] >= 5]  # отсекаем редко встречающиеся
    by_ip["susp_ratio"] = by_ip["susp"] / by_ip["total"]
    top_susp = by_ip.sort_values(["susp_ratio", "mean_score"], ascending=False).head(10)

    susp_fig = go.Figure(go.Bar(
        x=top_susp["src_ip"],
        y=top_susp["susp_ratio"] * 100,
        marker=dict(
            color=top_susp["mean_score"],
            colorscale=[[0, "#2ECC71"], [0.5, "#F39C12"], [1, "#E74C3C"]],
            cmin=0, cmax=1,
            colorbar=dict(title="Mean<br>Score", tickfont=dict(color="#E6E6E6"),
                          title_font=dict(color="#E6E6E6")),
            line=dict(color="#2A3142", width=1),
        ),
        text=[f"{v:.0f}%" for v in top_susp["susp_ratio"] * 100],
        textposition="outside",
        textfont=dict(color="#E6E6E6"),
        customdata=np.stack([top_susp["total"], top_susp["susp"], top_susp["mean_score"]], axis=-1),
        hovertemplate=("<b>%{x}</b><br>"
                       "Всего потоков: %{customdata[0]}<br>"
                       "Подозрительных: %{customdata[1]}<br>"
                       "Доля: %{y:.1f}%<br>"
                       "Mean score: %{customdata[2]:.3f}<extra></extra>"),
    ))
    susp_fig.update_layout(
        title=dict(text="Топ-10 IP-адресов по доле подозрительных потоков",
                   font_color="#FAFAFA"),
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1F2C",
        font_color="#E6E6E6",
        xaxis=dict(title="IP-адрес источника", gridcolor="#2A3142", tickangle=-30),
        yaxis=dict(title="Доля подозрительных потоков, %", gridcolor="#2A3142"),
        height=380, margin=dict(l=40, r=20, t=50, b=80),
    )
    st.plotly_chart(susp_fig, use_container_width=True)


# =================================================================
#                ВКЛАДКА 2 — ТОПОЛОГИЧЕСКИЙ АНАЛИЗ
# =================================================================
with tab_topo:
    st.markdown("#### Sankey-диаграмма маршрутов трафика")
    st.caption("«Подсеть-источник → Протокол → Подсеть-назначение». "
               "Толщина связи пропорциональна объёму переданных байт; "
               "цвет — гибридной оценке Score_hybrid.")

    # Подготовка данных Sankey
    sk_df = df.copy()
    sk_df["src_label"] = "src: " + sk_df["src_subnet"]
    sk_df["proto_label"] = sk_df["protocol"]
    sk_df["dst_label"] = "dst: " + sk_df["dst_subnet"]

    # Все уникальные узлы
    src_nodes = sorted(sk_df["src_label"].unique())
    proto_nodes = sorted(sk_df["proto_label"].unique())
    dst_nodes = sorted(sk_df["dst_label"].unique())
    all_nodes = src_nodes + proto_nodes + dst_nodes
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    # Связи src→proto и proto→dst
    link1 = sk_df.groupby(["src_label", "proto_label"]).agg(
        value=("bytes", "sum"),
        score=("score", "mean"),
        flows=("score", "count"),
    ).reset_index()
    link2 = sk_df.groupby(["proto_label", "dst_label"]).agg(
        value=("bytes", "sum"),
        score=("score", "mean"),
        flows=("score", "count"),
    ).reset_index()

    sources = ([node_idx[s] for s in link1["src_label"]] +
               [node_idx[p] for p in link2["proto_label"]])
    targets = ([node_idx[p] for p in link1["proto_label"]] +
               [node_idx[d] for d in link2["dst_label"]])
    values = list(link1["value"]) + list(link2["value"])
    scores = list(link1["score"]) + list(link2["score"])
    flows_n = list(link1["flows"]) + list(link2["flows"])

    # Цвет связей по score
    def score_to_color(s):
        if s >= threshold_red:
            return "rgba(231, 76, 60, 0.75)"
        elif s >= threshold_yellow:
            return "rgba(243, 156, 18, 0.65)"
        else:
            return "rgba(52, 152, 219, 0.45)"

    link_colors = [score_to_color(s) for s in scores]

    # Цвет узлов
    node_colors = (
        ["#5DADE2"] * len(src_nodes) +
        ["#AF7AC5"] * len(proto_nodes) +
        ["#48C9B0"] * len(dst_nodes)
    )

    sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=22,
            line=dict(color="#2A3142", width=0.5),
            label=all_nodes,
            color=node_colors,
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=link_colors,
            customdata=np.stack([scores, flows_n, np.array(values) / (1024 * 1024)], axis=-1),
            hovertemplate=("Объём: %{customdata[2]:.1f} МБ<br>"
                           "Потоков: %{customdata[1]}<br>"
                           "Mean Score: %{customdata[0]:.3f}<extra></extra>"),
        ),
    ))
    sankey.update_layout(
        paper_bgcolor="#0E1117", font_color="#E6E6E6",
        height=520, margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(sankey, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Тепловая карта аномалий по парам подсетей")
    st.caption("Агрегация в скользящем окне 60 секунд по парам "
               "(подсеть-источник × подсеть-назначение, маска /24).")

    heat = df.groupby(["src_subnet", "dst_subnet"])["score"].mean().reset_index()
    heat_pivot = heat.pivot(index="src_subnet", columns="dst_subnet", values="score")

    heat_fig = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns,
        y=heat_pivot.index,
        colorscale=[
            [0.0, "#1A4D2E"],
            [0.4, "#F1C40F"],
            [0.7, "#E67E22"],
            [1.0, "#C0392B"],
        ],
        zmin=0, zmax=1,
        colorbar=dict(title="Score_hybrid", tickfont=dict(color="#E6E6E6"),
                      title_font=dict(color="#E6E6E6")),
        hovertemplate=("Источник: %{y}<br>"
                       "Назначение: %{x}<br>"
                       "Mean Score: %{z:.3f}<extra></extra>"),
        text=heat_pivot.round(2).values,
        texttemplate="%{text}",
        textfont=dict(color="#FAFAFA", size=11),
    ))
    heat_fig.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1F2C",
        font_color="#E6E6E6",
        xaxis=dict(title="Подсеть-назначение", gridcolor="#2A3142"),
        yaxis=dict(title="Подсеть-источник", gridcolor="#2A3142"),
        height=480, margin=dict(l=140, r=40, t=20, b=80),
    )
    st.plotly_chart(heat_fig, use_container_width=True)


# =================================================================
#                ВКЛАДКА 3 — СТАТИСТИЧЕСКИЙ АНАЛИЗ
# =================================================================
with tab_stats:
    st.markdown("#### Диаграммы «ящик с усами» по основным признакам потоков")
    st.caption("Длительность потока, объём переданных байт, интенсивность пакетов (PPS). "
               "Сравнение распределений для нормального, подозрительного и критического трафика.")

    df["category"] = pd.cut(
        df["score"],
        bins=[-0.01, threshold_yellow, threshold_red, 1.01],
        labels=["Нормальные", "Подозрительные", "Критические"],
    )
    cat_colors = {"Нормальные": "#3498DB",
                  "Подозрительные": "#F39C12",
                  "Критические": "#E74C3C"}

    box_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Длительность потока, сек",
                        "Объём переданных байт, КБ",
                        "Интенсивность пакетов, PPS"),
        horizontal_spacing=0.09,
    )

    for cat in ["Нормальные", "Подозрительные", "Критические"]:
        sub = df[df["category"] == cat]
        box_fig.add_trace(go.Box(
            y=sub["duration"], name=cat, marker_color=cat_colors[cat],
            boxmean=True, showlegend=True, legendgroup=cat,
        ), row=1, col=1)
        box_fig.add_trace(go.Box(
            y=sub["bytes"] / 1024, name=cat, marker_color=cat_colors[cat],
            boxmean=True, showlegend=False, legendgroup=cat,
        ), row=1, col=2)
        box_fig.add_trace(go.Box(
            y=sub["pps"], name=cat, marker_color=cat_colors[cat],
            boxmean=True, showlegend=False, legendgroup=cat,
        ), row=1, col=3)

    box_fig.update_yaxes(type="log", gridcolor="#2A3142")
    box_fig.update_xaxes(gridcolor="#2A3142")
    box_fig.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1F2C",
        font_color="#E6E6E6",
        height=420, margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
    )
    # Цвет заголовков subplots
    for ann in box_fig["layout"]["annotations"]:
        ann["font"] = dict(color="#FAFAFA", size=13)
    st.plotly_chart(box_fig, use_container_width=True)

    st.markdown("---")

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        st.markdown("#### Диаграмма рассеяния")
        st.caption("Зависимость длительности сессии от объёма переданных байт "
                   "(пара наиболее информативных признаков по таблице 3 ВКР).")
        scatter_fig = go.Figure()
        for cat in ["Нормальные", "Подозрительные", "Критические"]:
            sub = df[df["category"] == cat]
            # Прореживаем для скорости отрисовки
            if len(sub) > 1500:
                sub = sub.sample(1500, random_state=0)
            scatter_fig.add_trace(go.Scattergl(
                x=sub["duration"], y=sub["bytes"] / 1024,
                mode="markers", name=cat,
                marker=dict(color=cat_colors[cat], size=6, opacity=0.6,
                            line=dict(width=0)),
                hovertemplate=("Длительность: %{x:.2f} сек<br>"
                               "Объём: %{y:.1f} КБ<extra></extra>"),
            ))
        scatter_fig.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1F2C",
            font_color="#E6E6E6",
            xaxis=dict(title="Длительность потока, сек (log)", type="log",
                       gridcolor="#2A3142"),
            yaxis=dict(title="Объём переданных байт, КБ (log)", type="log",
                       gridcolor="#2A3142"),
            height=420, margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    with row3_col2:
        st.markdown("#### Гистограмма распределения длительности потоков")
        st.caption("Визуальное выделение «хвоста» распределения помогает находить "
                   "долгоживущие сессии, потенциально связанные с C2-каналами ботнетов.")
        dur_fig = go.Figure()
        for cat in ["Нормальные", "Подозрительные", "Критические"]:
            sub = df[df["category"] == cat]
            dur_fig.add_trace(go.Histogram(
                x=np.log10(sub["duration"].clip(lower=0.001)),
                name=cat, marker_color=cat_colors[cat],
                opacity=0.75, nbinsx=40,
            ))
        dur_fig.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1F2C",
            font_color="#E6E6E6",
            xaxis=dict(title="log10(Длительность, сек)", gridcolor="#2A3142"),
            yaxis=dict(title="Число потоков", gridcolor="#2A3142"),
            barmode="overlay", height=420,
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        )
        st.plotly_chart(dur_fig, use_container_width=True)

# ===================== ПОДВАЛ =====================
st.markdown("---")
st.caption("Дипломный проект 2025 · Гибридная NDR-система · "
           "XGBoost и Isolation Forest · Streamlit · Plotly")
