# src/visualization/analyzer_full.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import xgboost as xgb   # ← ЭТО БЫЛО НЕ ХВАТИЛО!
import sys

# Фикс импорта при запуске скрипта напрямую
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

from src.preprocessing.feature_extractor import extract_flow_level_features as extract_features

# ====================== НАСТРОЙКИ ======================
st.set_page_config(
    page_title="Network Anomaly Visualizer",
    page_icon="Network",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Network Anomaly Visualizer 2025")
st.markdown("### Гибридная NIDS: XGBoost + Isolation Forest + поведенческие фичи")

# ====================== ЗАГРУЗКА МОДЕЛЕЙ ======================
@st.cache_resource
def load_models():
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("models/xgboost_model.json")
        iso_model = joblib.load("models/isolation_forest.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_list = joblib.load("models/feature_list.pkl")
        return xgb_model, iso_model, scaler, feature_list
    except Exception as e:
        st.error(f"Не удалось загрузить модели: {e}")
        st.error("Запусти: python scripts/train_models.py")
        st.stop()

xgb_model, iso_model, scaler, feature_list = load_models()

# ====================== САЙДБАР + АВТО-ЗАГРУЗКА ======================
st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Выбери CSV с потоками", type=["csv"])

# ИНИЦИАЛИЗИРУЕМ df_raw = None
df_raw = None

# 1. Ручная загрузка
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Загружено {len(df_raw):,} строк")

# 2. Кнопка демо
elif st.sidebar.button("Демо: смешанный трафик"):
    demo_path = Path("data/demo/demo_mixed_traffic.csv")
    if demo_path.exists():
        df_raw = pd.read_csv(demo_path)
        st.sidebar.success("Демо загружено!")
    else:
        st.sidebar.error("Файл демо не найден")

# 3. АВТО-ЗАГРУЗКА БОЛЬШОГО ДАТАСЕТА (главное!)
elif Path("data/processed/my_traffic_features.csv").exists():
    df_raw = pd.read_csv("data/processed/my_traffic_features.csv")
    st.sidebar.success("Автозагрузка: DDoS-датасет (225k потоков)")

else:
    st.sidebar.info("Загрузи файл или используй демо")
# ====================== ПРЕДСКАЗАНИЯ ======================
if df_raw is not None:
    with st.spinner("Анализ трафика и предсказание..."):
        df = extract_features(df_raw.copy())

        # Подгоняем под ожидаемые моделью фичи
        X = pd.DataFrame(0.0, index=df.index, columns=feature_list)
        for col in feature_list:
            if col in df.columns:
                X[col] = df[col]

        X_scaled = scaler.transform(X)

        # Предсказания
        xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
        iso_raw = -iso_model.decision_function(X_scaled)
        iso_score = (iso_raw - iso_raw.min()) / (iso_raw.ptp() + 1e-12)
        hybrid_score = np.maximum(xgb_proba, iso_score)

        df["anomaly_score"] = hybrid_score
        df["prediction"] = (hybrid_score >= 0.5).astype(int)
        df["attack_type"] = df["prediction"].map({0: "Normal", 1: "Attack"})

    st.success(f"Готово! Обнаружено атак: {df['prediction'].sum():,}")

    # ====================== ВИЗУАЛИЗАЦИИ ======================
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Аномалии во времени — общая картина сети")

        # Время
        time_col = "timestamp_start" if "timestamp_start" in df.columns else "timestamp"
        if time_col not in df.columns:
            df["timestamp"] = range(len(df))
            time_col = "timestamp"

        df_plot = df.copy()
        df_plot["time"] = pd.to_datetime(df_plot[time_col], unit='s', errors='coerce')
        df_plot = df_plot.sort_values("time")

        fig = go.Figure()

        # 1. СНАЧАЛА — ВСЕ ТОЧКИ (серый фон)
        fig.add_trace(go.Scattergl(
            x=df_plot["time"],
            y=df_plot["anomaly_score"],
            mode='markers',
            marker=dict(color='lightgray', size=4, opacity=0.4),
            name='Все потоки',
            hoverinfo='skip'
        ))

        # 2. ПОТОМ — АТАКИ ПОВЕРХ ВСЕГО (красные ракеты, всегда видно!)
        attack = df_plot[df_plot["anomaly_score"] >= 0.80]
        if len(attack) > 0:
            fig.add_trace(go.Scattergl(
                x=attack["time"],
                y=attack["anomaly_score"],
                mode='markers',
                name=f'АТАКА! ({len(attack):,} шт.)',
                marker=dict(
                    color='#ff0000',
                    size=18,
                    symbol='triangle-up',
                    line=dict(color='darkred', width=4),
                    opacity=1.0
                ),
                text=[f"<b>DDoS АТАКА</b><br>"
                      f"IP: {r.get('src_ip', '?')} → {r.get('dst_ip', '?')}<br>"
                      f"pps: {r.get('pps', 0):,.0f}<br>"
                      f"Score: {r['anomaly_score']:.4f}"
                      for _, r in attack.iterrows()],
                hovertemplate="%{text}<extra></extra>"
            ))
        else:
            st.warning("Атак не найдено (проверь порог)")

        # Порог
        fig.add_hline(y=0.80, line=dict(color="red", width=3, dash="dash"),
                      annotation_text=" Порог 0.80", annotation_position="top right")

        fig.update_layout(
            title="Гибридный скор аномалий — DDoS-атака в реальном времени",
            xaxis_title="Время",
            yaxis_title="Anomaly Score",
            yaxis=dict(range=[0, 1.05]),
            height=720,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
        )

        st.plotly_chart(fig, use_container_width=True)

        # Добавим статистику под графиком
        st.success(f"Обнаружено атак: **{len(attack):,}** из {len(df_plot):,} потоков")

    with col2:
        st.subheader("Статистика")

        # === Метрики сверху ===
        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            st.metric("Всего потоков", f"{len(df):,}")
        with colm2:
            attacks_count = df["prediction"].sum()
            st.metric("Обнаружено атак", f"{attacks_count:,}")
        with colm3:
            st.metric("Доля атак", f"{df['prediction'].mean():.2%}")

        # === КРУТОЙ DONUT CHART ПО ПРОТОКОЛАМ (ТОЧНО КАК НА КАРТИНКЕ) ===
        if "protocol" in df.columns and attacks_count > 0:
            # Считаем только атаки!
            attack_df = df[df["prediction"] == 1]

            # Приводим протоколы к красивому виду
            protocol_map = {
                'TCP': 'TCP/UDP',
                'UDP': 'TCP/UDP',
                'HTTP': 'HTTP/HTTPS',
                'HTTPS': 'HTTP/HTTPS',
                'DNS': 'DNS',
                'ICMP': 'Другие',
                'IGMP': 'Другие',
                'OTHER': 'Другие',
                '': 'Другие'
            }
            attack_df = attack_df.copy()
            attack_df["proto_group"] = attack_df["protocol"].astype(str).str.upper().map(protocol_map).fillna("Другие")

            # Считаем распределение
            proto_counts = attack_df["proto_group"].value_counts()
            total_attacks = proto_counts.sum()

            # Если всё одного типа — покажем красивую заглушку
            if len(proto_counts) == 1:
                fig_proto = go.Figure(go.Pie(
                    labels=["Обнаруженные атаки"],
                    values=[100],
                    hole=0.5,
                    marker_colors=["#1E88E5"],
                    textinfo="percent+label",
                    hoverinfo="label+percent",
                    showlegend=False
                ))
                fig_proto.update_layout(
                    title_text="Все атаки<br>по одному протоколу",
                    annotations=[dict(text=f"{total_attacks:,}<br>атак", x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
            else:
                # Основной график — как на твоей картинке!
                colors = {
                    "HTTP/HTTPS": "#1E88E5",
                    "TCP/UDP": "#E91E63",
                    "DNS": "#4CAF50",
                    "Другие": "#FF9800"
                }
                fig_proto = go.Figure(go.Pie(
                    labels=proto_counts.index,
                    values=proto_counts.values,
                    hole=0.5,  # ← дырка посередине
                    marker_colors=[colors.get(p, "#9E9E9E") for p in proto_counts.index],
                    textinfo="percent+label",
                    hoverinfo="label+percent+value",
                    sort=False
                ))

                # Большой текст в центре (как на фото!)
                main_proto = proto_counts.idxmax()
                main_percent = (proto_counts.max() / total_attacks * 100)
                fig_proto.update_layout(
                    title_text="Атаки по протоколам",
                    annotations=[
                        dict(
                            text=f"{main_percent:.0f}%",
                            x=0.5, y=0.55,
                            font_size=32,
                            font=dict(color=colors.get(main_proto, "#1E88E5")),
                            showarrow=False
                        ),
                        dict(
                            text=main_proto,
                            x=0.5, y=0.42,
                            font_size=14,
                            showarrow=False
                        ),
                        # Иконка щита в центре (чисто для красоты)
                        dict(
                            text="",
                            x=0.5, y=0.5,
                            font_size=40,
                            showarrow=False
                        )
                    ]
                )

            st.plotly_chart(fig_proto, use_container_width=True)

        else:
            # Если нет атак или нет протоколов — просто топ IP
            if "src_ip" in df.columns and attacks_count > 0:
                top_ips = df[df["prediction"] == 1]["src_ip"].value_counts().head(8)
                if len(top_ips) > 0:
                    fig_pie = px.pie(
                        values=top_ips.values,
                        names=top_ips.index,
                        title="Топ атакующих IP",
                        color_discrete_sequence=px.colors.sequential.Reds,
                        hole=0.4
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

    # Важность  призанков(XGBoost)
    # Важность признаков (XGBoost)
    st.subheader("Важность признаков (XGBoost)")
    imp = pd.DataFrame({
        "feature": feature_list,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

    fig_imp = px.bar(
        imp,
        x="importance",
        y="feature",
        orientation="h",
        title="Топ-15 самых важных признаков",
        color="importance",
        color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(height=600)
    st.plotly_chart(fig_imp, use_container_width=True)



    # Скачать
    csv = df.to_csv(index=False).encode()
    st.download_button("Скачать результаты", csv, "anomalies_detected.csv", "text/csv")

else:
    st.info("Загрузи свой CSV или нажми кнопку демо")
    st.image("https://i.imgur.com/8z2lN8k.png", use_column_width=True)

st.markdown("---")
st.caption("Дипломная работа 2025 • Гибридная NIDS с поведенческим анализом")