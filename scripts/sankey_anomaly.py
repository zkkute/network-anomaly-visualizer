"""
Sankey-диаграмма потоков трафика с цветовой индикацией аномальности связей.

Структура графа: подсеть-источник /24  ->  протокол  ->  подсеть-назначения /24
- Толщина связи: пропорциональна суммарному объёму переданных байт.
- Цвет связи: средняя гибридная оценка Score_hybrid (зелёный -> жёлтый -> красный).

Источник данных: data/live/flows_live.csv (формат как в real_live/realtime_engine.py)
Обязательные колонки: src_ip, dst_ip, score, bytes
Опциональные: protocol (если нет — будет TCP/UDP по dst_port-эвристике или 'OTHER')

Выходные файлы:
  reports/figures/sankey_anomaly.png   — для вставки в Word (300 DPI)
  reports/figures/sankey_anomaly.html  — интерактивный, для защиты

Запуск:
  python scripts/sankey_anomaly.py
  python scripts/sankey_anomaly.py --input data/processed/my_traffic_features.csv
  python scripts/sankey_anomaly.py --demo    # синтетические данные
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ===================== КОНФИГ =====================

DEFAULT_INPUT = Path("data/live/flows_live.csv")
OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT_DIR / "sankey_anomaly.png"
HTML_PATH = OUT_DIR / "sankey_anomaly.html"

# Маска подсети (24 = 192.168.10.0/24)
SUBNET_MASK_BITS = 24

# Сколько верхних подсетей оставлять
TOP_N_SRC = 6
TOP_N_DST = 6

# Порог для пометки связи как аномальной в hover
ANOMALY_THRESHOLD = 0.5


# ===================== ВСПОМОГАТЕЛЬНОЕ =====================

def ip_to_subnet(ip, prefix_bits=SUBNET_MASK_BITS):
    """192.168.10.5 -> 192.168.10.0/24"""
    try:
        parts = str(ip).strip().split(".")
        if len(parts) != 4:
            return "unknown"
        octets = [int(p) for p in parts]
        keep = prefix_bits // 8
        masked = octets[:keep] + [0] * (4 - keep)
        return ".".join(str(o) for o in masked) + f"/{prefix_bits}"
    except (ValueError, AttributeError):
        return "unknown"


def score_to_rgba(score, alpha=0.55):
    """Цвет связи: зелёный (0) -> жёлтый (0.5) -> красный (1)."""
    s = max(0.0, min(1.0, float(score)))
    if s < 0.5:
        t = s / 0.5
        r = int(46 + (255 - 46) * t)
        g = int(184 + (193 - 184) * t)
        b = int(46 + (7 - 46) * t)
    else:
        t = (s - 0.5) / 0.5
        r = int(255 + (220 - 255) * t)
        g = int(193 + (38 - 193) * t)
        b = int(7 + (38 - 7) * t)
    return f"rgba({r},{g},{b},{alpha})"


def detect_protocol(df):
    """Колонка protocol -> используем. Иначе угадываем по dst_port."""
    if "protocol" in df.columns and df["protocol"].notna().any():
        return df["protocol"].astype(str).str.upper().replace({"": "OTHER"}).fillna("OTHER")
    if "dst_port" in df.columns:
        def guess(p):
            try:
                p = int(p)
            except (ValueError, TypeError):
                return "OTHER"
            if p in (80, 8080, 8000): return "HTTP"
            if p in (443, 8443):       return "HTTPS"
            if p == 53:                return "DNS"
            if p in (22, 23):          return "SSH/Telnet"
            if p in (21, 20):          return "FTP"
            if p in (25, 465, 587):    return "SMTP"
            return "TCP/UDP"

        return df["dst_port"].apply(guess)
    return pd.Series(["TCP/UDP"] * len(df), index=df.index)


def keep_top_n(series, top_n, other_label="other"):
    """Top-N по частоте, остальное -> other_label."""
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), other_label)


def generate_demo_flows(n=4000, seed=42):
    """Синтетические данные для теста."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        if rng.random() < 0.8:  # фоновой трафик
            src = f"192.168.{rng.integers(10, 14)}.{rng.integers(1, 254)}"
            dst = f"203.0.{rng.integers(110, 114)}.{rng.integers(1, 254)}"
            score = float(rng.beta(1.2, 8))
            bytes_v = int(rng.lognormal(7.5, 1.5))
        else:  # аномалия — эксфильтрация из 192.168.10.0/24
            src = f"192.168.10.{rng.integers(1, 254)}"
            dst = f"203.0.113.{rng.integers(1, 254)}"
            score = float(rng.uniform(0.7, 0.98))
            bytes_v = int(rng.lognormal(10.5, 0.8))
        proto = rng.choice(["TCP/UDP", "HTTPS", "HTTP", "DNS"], p=[0.5, 0.3, 0.15, 0.05])
        rows.append({"src_ip": src, "dst_ip": dst, "protocol": proto,
                     "score": score, "bytes": bytes_v})
    return pd.DataFrame(rows)


# ===================== ОСНОВНАЯ ЛОГИКА =====================

def build_sankey(df):
    df = df.copy()

    # 1. Подсети и протоколы
    df["src_subnet"] = df["src_ip"].apply(ip_to_subnet)
    df["dst_subnet"] = df["dst_ip"].apply(ip_to_subnet)
    df["proto"] = detect_protocol(df)

    # 2. Bytes (с fallback на альтернативные имена)
    if "bytes" not in df.columns:
        for cand in ("Tot Bytes", "fwd_bytes", "Flow Bytes/s"):
            if cand in df.columns:
                df["bytes"] = pd.to_numeric(df[cand], errors="coerce").fillna(0)
                break
        else:
            df["bytes"] = 1.0

    # 3. Score (с fallback)
    if "score" not in df.columns:
        for cand in ("anomaly_score", "Score_hybrid", "hybrid_score"):
            if cand in df.columns:
                df["score"] = pd.to_numeric(df[cand], errors="coerce").fillna(0)
                break
        else:
            print("[!] Колонка score не найдена — связи будут окрашены нейтрально")
            df["score"] = 0.3

    # 4. Top-N подсетей
    df["src_subnet"] = keep_top_n(df["src_subnet"], TOP_N_SRC, "src_other")
    df["dst_subnet"] = keep_top_n(df["dst_subnet"], TOP_N_DST, "dst_other")

    # 5. Агрегация связей
    src_to_proto = (df.groupby(["src_subnet", "proto"])
                    .agg(bytes_sum=("bytes", "sum"),
                         score_mean=("score", "mean"),
                         flow_count=("score", "count"))
                    .reset_index())
    proto_to_dst = (df.groupby(["proto", "dst_subnet"])
                    .agg(bytes_sum=("bytes", "sum"),
                         score_mean=("score", "mean"),
                         flow_count=("score", "count"))
                    .reset_index())

    # 6. Узлы
    src_nodes = sorted(df["src_subnet"].unique())
    proto_nodes = sorted(df["proto"].unique())
    dst_nodes = sorted(df["dst_subnet"].unique())
    all_nodes = src_nodes + proto_nodes + dst_nodes
    node_idx = {name: i for i, name in enumerate(all_nodes)}

    node_colors = (["rgba(70,130,180,0.85)"] * len(src_nodes) +
                   ["rgba(120,120,120,0.85)"] * len(proto_nodes) +
                   ["rgba(180,90,90,0.85)"] * len(dst_nodes))

    # 7. Связи
    sources, targets, values, link_colors, hover = [], [], [], [], []

    for _, row in src_to_proto.iterrows():
        sources.append(node_idx[row["src_subnet"]])
        targets.append(node_idx[row["proto"]])
        values.append(float(row["bytes_sum"]))
        link_colors.append(score_to_rgba(row["score_mean"]))
        flag = " (аномалия)" if row["score_mean"] >= ANOMALY_THRESHOLD else ""
        hover.append(
            f"{row['src_subnet']} → {row['proto']}<br>"
            f"Потоков: {int(row['flow_count'])}<br>"
            f"Объём: {row['bytes_sum']:,.0f} байт<br>"
            f"Средн. Score: {row['score_mean']:.3f}{flag}"
        )

    for _, row in proto_to_dst.iterrows():
        sources.append(node_idx[row["proto"]])
        targets.append(node_idx[row["dst_subnet"]])
        values.append(float(row["bytes_sum"]))
        link_colors.append(score_to_rgba(row["score_mean"]))
        flag = " (аномалия)" if row["score_mean"] >= ANOMALY_THRESHOLD else ""
        hover.append(
            f"{row['proto']} → {row['dst_subnet']}<br>"
            f"Потоков: {int(row['flow_count'])}<br>"
            f"Объём: {row['bytes_sum']:,.0f} байт<br>"
            f"Средн. Score: {row['score_mean']:.3f}{flag}"
        )

    # 8. Sankey
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=22,
            line=dict(color="rgba(0,0,0,0.4)", width=0.8),
            label=all_nodes,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    )])

    fig.update_layout(
        title=dict(
            text="Sankey-диаграмма потоков трафика<br>"
                 "<sub>Толщина — объём байт; цвет — средняя гибридная оценка Score_hybrid "
                 "(зелёный → красный)</sub>",
            x=0.5, xanchor="center", font=dict(size=15),
        ),
        font=dict(size=12, family="Arial"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=80, b=20),
        width=1400,
        height=750,
    )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Путь к CSV с потоками")
    parser.add_argument("--demo", action="store_true",
                        help="Использовать синтетические данные")
    args = parser.parse_args()

    if args.demo:
        print("[i] Режим демо: генерирую синтетические потоки")
        df = generate_demo_flows()
    else:
        path = Path(args.input)
        if not path.exists():
            print(f"[!] Файл не найден: {path}")
            print("    Запустите с флагом --demo для синтетических данных")
            sys.exit(1)
        df = pd.read_csv(path)
        print(f"[i] Загружено {len(df):,} потоков из {path}")

    fig = build_sankey(df)

    fig.write_html(HTML_PATH, include_plotlyjs="cdn")
    print(f"[+] Сохранено: {HTML_PATH}")

    try:
        fig.write_image(PNG_PATH, width=1400, height=750, scale=2)
        print(f"[+] Сохранено: {PNG_PATH}")
    except Exception as e:
        print(f"[!] Не удалось сохранить PNG ({e}).")
        print("    Установите: pip install -U kaleido")


if __name__ == "__main__":
    main()