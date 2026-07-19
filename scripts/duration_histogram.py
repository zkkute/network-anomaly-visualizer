"""
Гистограмма распределения длительности сетевых потоков.

Цель: показать основной массив коротких запросов и хвост долгоживущих соединений
(потенциально — каналов C&C / эксфильтрация).

Источник данных:
  data/processed/my_traffic_features.csv  (есть колонка duration)
  ИЛИ data/live/flows_live.csv
  ИЛИ data/demo/flows_demo_for_histogram.csv

Что строится:
  - Логарифмическая ось X (иначе хвост не виден)
  - Ручные log-bins через numpy + рисование через go.Bar — полный контроль над масштабом
  - Подсвечены медиана (зелёный) и p95 (оранжевый)
  - Норма и аномалии — разными цветами (overlay)
  - Сводка статистики в правом углу

Выходные файлы:
  reports/figures/duration_histogram.png   — для Word (300 DPI)
  reports/figures/duration_histogram.html  — интерактивный

Запуск:
  python scripts/duration_histogram.py
  python scripts/duration_histogram.py --input data/demo/flows_demo_for_histogram.csv
  python scripts/duration_histogram.py --demo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ===================== КОНФИГ =====================

DEFAULT_INPUT = Path("data/processed/my_traffic_features.csv")
OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT_DIR / "duration_histogram.png"
HTML_PATH = OUT_DIR / "duration_histogram.html"

ANOMALY_THRESHOLD = 0.5
N_BINS = 50


# ===================== ВСПОМОГАТЕЛЬНОЕ =====================

def find_duration_column(df):
    """Ищет колонку длительности. Возвращает (имя_колонки, scale_к_секундам)."""
    candidates = [
        ("duration", 1.0),
        ("Flow Duration", 1e-6),  # CIC-IDS: мкс
        ("Flow_Duration", 1e-6),
        ("flow_duration", 1.0),
    ]
    for name, scale in candidates:
        if name in df.columns:
            return name, scale
    return None, None


def detect_attack_mask(df):
    """Булева маска атаки или None."""
    if "label" in df.columns:
        lab = df["label"]
        if pd.api.types.is_numeric_dtype(lab):
            return lab.astype(int) == 1
        normal = {"normal", "benign", "0", "ok", ""}
        return ~lab.astype(str).str.strip().str.lower().isin(normal)
    if "score" in df.columns:
        return pd.to_numeric(df["score"], errors="coerce").fillna(0) >= ANOMALY_THRESHOLD
    if "anomaly_score" in df.columns:
        return pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0) >= ANOMALY_THRESHOLD
    return None


def generate_demo_durations(n=20000, seed=42):
    """Бимодальное распределение для теста."""
    rng = np.random.default_rng(seed)
    short = rng.lognormal(mean=0.3, sigma=1.2, size=int(n * 0.92))
    long_tail = rng.lognormal(mean=5.0, sigma=1.0, size=int(n * 0.08))
    durations = np.concatenate([short, long_tail])
    labels = np.concatenate([np.zeros(len(short)), np.ones(len(long_tail))]).astype(int)
    return pd.DataFrame({"duration": durations, "label": labels})


def histogram_log_bins(values, n_bins, x_min, x_max):
    """Считает гистограмму на лог-шкале. Возвращает (centers, counts, widths)."""
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)
    # Центры в лог-пространстве — для красивого выравнивания столбиков
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    widths = bin_edges[1:] - bin_edges[:-1]
    return centers, counts, widths


# ===================== ОСНОВНАЯ ЛОГИКА =====================

def build_histogram(df):
    # 1. Колонка длительности
    col, scale = find_duration_column(df)
    if col is None:
        raise ValueError("Не найдена колонка duration / Flow Duration. "
                         "Доступные колонки: " + ", ".join(df.columns[:20]))

    dur_series = pd.to_numeric(df[col], errors="coerce").fillna(0) * scale
    valid_idx = dur_series > 0
    dur_series = dur_series[valid_idx]
    df_clean = df.loc[valid_idx]

    if len(dur_series) == 0:
        raise ValueError("После очистки не осталось положительных значений duration.")

    dur = dur_series.values
    print(f"[i] Использую колонку '{col}' (scale={scale}), валидных значений: {len(dur):,}")

    # 2. Статистика
    p50 = np.median(dur)
    p95 = np.percentile(dur, 95)
    p99 = np.percentile(dur, 99)
    mean_v = float(dur.mean())
    max_v = float(dur.max())
    min_v = float(dur.min())

    # 3. ГРАНИЦЫ ОСИ X — самое важное!
    # Берём реальный диапазон данных с небольшим запасом, а не степени 10
    # Это и устраняет «пустоту до 1Т секунд» из предыдущей версии
    x_min = max(min_v, 1e-3)  # не уходим в микросекунды зря
    x_max = max_v * 1.2       # 20% запаса справа от max — чтобы хвост не упирался в край

    # 4. Лог-бины
    attack_mask = detect_attack_mask(df_clean)
    if attack_mask is not None:
        attack_mask = attack_mask.values
        normal_durs = dur[~attack_mask]
        attack_durs = dur[attack_mask]
    else:
        normal_durs = dur
        attack_durs = np.array([])

    fig = go.Figure()

    if len(attack_durs) > 0 and len(normal_durs) > 0:
        centers_n, counts_n, widths_n = histogram_log_bins(normal_durs, N_BINS, x_min, x_max)
        centers_a, counts_a, widths_a = histogram_log_bins(attack_durs, N_BINS, x_min, x_max)

        fig.add_trace(go.Bar(
            x=centers_n, y=counts_n, width=widths_n,
            name=f"Норма ({len(normal_durs):,})",
            marker=dict(color="rgba(70,130,180,0.75)",
                        line=dict(color="rgba(70,130,180,1)", width=0.5)),
            hovertemplate="Длительность: %{x:.3f} с<br>Норма: %{y}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=centers_a, y=counts_a, width=widths_a,
            name=f"Аномалии ({len(attack_durs):,})",
            marker=dict(color="rgba(220,38,38,0.75)",
                        line=dict(color="rgba(220,38,38,1)", width=0.5)),
            hovertemplate="Длительность: %{x:.3f} с<br>Аномалии: %{y}<extra></extra>",
        ))
        fig.update_layout(barmode="overlay")
    else:
        centers, counts, widths = histogram_log_bins(dur, N_BINS, x_min, x_max)
        fig.add_trace(go.Bar(
            x=centers, y=counts, width=widths,
            name="Все потоки",
            marker=dict(color="rgba(70,130,180,0.85)",
                        line=dict(color="rgba(70,130,180,1)", width=0.5)),
            hovertemplate="Длительность: %{x:.3f} с<br>Потоков: %{y}<extra></extra>",
        ))

    # 5. Вертикальные линии для медианы и p95
    fig.add_vline(x=p50, line=dict(color="green", width=2, dash="dot"),
                  annotation_text=f"медиана = {p50:.2f} с",
                  annotation_position="top",
                  annotation=dict(font=dict(size=11, color="green")))
    fig.add_vline(x=p95, line=dict(color="orange", width=2, dash="dash"),
                  annotation_text=f"p95 = {p95:.2f} с",
                  annotation_position="top",
                  annotation=dict(font=dict(size=11, color="orange")))

    # 6. Сводка статистики
    stats_text = (
        f"<b>Статистика (сек)</b><br>"
        f"Всего потоков: {len(dur):,}<br>"
        f"Медиана: {p50:.3f}<br>"
        f"Среднее: {mean_v:.3f}<br>"
        f"p95: {p95:.3f}<br>"
        f"p99: {p99:.3f}<br>"
        f"Max: {max_v:.1f}"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.97,
        text=stats_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(0,0,0,0.3)",
        borderwidth=1,
        font=dict(size=11, family="Arial"),
    )

    # 7. Оформление — ЯВНО задаём range оси X в log-координатах
    fig.update_layout(
        title=dict(
            text="Гистограмма распределения длительности сетевых потоков<br>"
                 "<sub>Логарифмическая шкала по X; хвост справа — долгоживущие сессии</sub>",
            x=0.5, xanchor="center", font=dict(size=15),
        ),
        xaxis=dict(
            title="Длительность потока, секунды (log scale)",
            type="log",
            # КЛЮЧЕВОЕ: явно ограничиваем диапазон оси, иначе plotly расширит до 10^N
            range=[np.log10(x_min), np.log10(x_max)],
            showgrid=True, gridcolor="rgba(0,0,0,0.08)",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Количество потоков",
            showgrid=True, gridcolor="rgba(0,0,0,0.08)",
            tickfont=dict(size=11),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        width=1300,
        height=650,
        legend=dict(x=0.02, y=0.97, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.2)", borderwidth=1),
        margin=dict(l=70, r=40, t=80, b=70),
        bargap=0.02,  # практически без зазоров между столбиками
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
        print("[i] Режим демо: генерирую синтетические длительности")
        df = generate_demo_durations()
    else:
        path = Path(args.input)
        if not path.exists():
            print(f"[!] Файл не найден: {path}")
            print("    Запустите с флагом --demo для синтетических данных")
            sys.exit(1)
        df = pd.read_csv(path)
        print(f"[i] Загружено {len(df):,} строк из {path}")

    fig = build_histogram(df)

    fig.write_html(HTML_PATH, include_plotlyjs="cdn")
    print(f"[+] Сохранено: {HTML_PATH}")

    try:
        fig.write_image(PNG_PATH, width=1300, height=650, scale=2)
        print(f"[+] Сохранено: {PNG_PATH}")
    except Exception as e:
        print(f"[!] Не удалось сохранить PNG ({e}).")
        print("    Установите: pip install -U kaleido")


if __name__ == "__main__":
    main()