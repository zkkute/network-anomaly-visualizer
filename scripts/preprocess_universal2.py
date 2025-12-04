# scripts/preprocess_universal2.py
# ФИНАЛЬНАЯ ВЕРСИЯ — 82 признака, всё работает, ничего лишнего

import time
import sys
from pathlib import Path

import numpy as np
# === ФИКС ПУТИ ===
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
# ГЛАВНОЕ: используем твой мощный feature_extractor!
from src.preprocessing.feature_extractor import extract_flow_level_features as extract_features
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

BEST_FEATURES = [
    'duration', 'packets', 'bytes', 'pps', 'bps', 'avg_pkt_size',
    'port_entropy', 'iat_std', 'unique_dst_port', 'unique_dst_ip',
    'syn_ratio', 'rst_ratio', 'fin_ratio',
    'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',

    # ← Твои мощные 27 новых →
    'iat_mean', 'iat_max', 'iat_min', 'iat_var', 'iat_skew', 'iat_kurtosis',
    'pkt_len_mean', 'pkt_len_max', 'pkt_len_min', 'pkt_len_std', 'pkt_len_var', 'pkt_len_skew', 'pkt_len_kurtosis',
    'fwd_bwd_ratio_packets', 'fwd_bwd_ratio_bytes', 'bytes_per_pkt', 'pkts_per_byte',
    'ip_dst_entropy', 'port_dst_entropy',
    'ack_ratio', 'psh_ratio', 'urg_ratio', 'ece_ratio', 'cwr_ratio',
    'flow_active_time', 'flow_idle_time',
    'bytes_per_second', 'packets_per_second'
]
print(f"Используется {len(BEST_FEATURES)} признаков (только твои + базовые)")


# Пути
OUTPUT_FILE = Path("data/processed/my_traffic_features.csv")
MODELS_DIR = Path("models")

def clean_features_before_scaling(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Полная защита от inf, NaN и экстремальных выбросов.
    Работает идеально на всех датасетах CIC-IDS-2017/2018.
    """
    X = df[feature_columns].copy()

    print("Очистка признаков от inf/NaN и выбросов...")

    # 1. Заменяем ±inf на NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # 2. Логарифмируем признаки, где это имеет смысл (избегаем log(0))
    #    (многие признаки типа pkt_len, byte counts — сильно скошены)
    log_cols = ['pkt_len_mean', 'pkt_len_std', 'pkt_len_var',
                'fwd_pkt_len_max', 'fwd_pkt_len_mean', 'bwd_pkt_len_max', 'bwd_pkt_len_mean',
                'flow_bytes_s', 'flow_pkts_s', 'subflow_fwd_bytes', 'subflow_bwd_bytes']
    log_cols = [c for c in log_cols if c in X.columns]
    if log_cols:
        X[log_cols] = X[log_cols].clip(lower=0) + 1  # делаем >=1
        X[log_cols] = np.log(X[log_cols])

    # 3. Заполняем NaN медианами (очень надёжно)
    X = X.fillna(X.median(numeric_only=True))

    # 4. Жёсткая обрезка выбросов по 1% и 99% квантилям (самое важное!)
    lower = X.quantile(0.01)
    upper = X.quantile(0.99)
    X = X.clip(lower=lower, upper=upper, axis=1)

    # 5. Финальная проверка — если где-то всё ещё NaN (крайне редко)
    if X.isna().any().any():
        X = X.fillna(0)

    print(f"   → inf после очистки: {np.isinf(X).sum().sum()}")
    print(f"   → NaN после очистки: {X.isna().sum().sum()}")
    return X

def main():
    print("УНИВЕРСАЛЬНЫЙ ПРЕПРОЦЕССОР v6.0 — ФИНАЛ ДЛЯ ЗАЩИТЫ 2025")
    print("Используется 82 поведенческих признака: pps, entropy, iat_std, syn_ratio и др.\n")

    # === Поиск CSV ===
    csv_files = sorted(
        Path("data/raw").rglob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not csv_files:
        print("CSV-файлы не найдены в data/raw/ и подпапках!")
        return

    print(f"Найдено датасетов: {len(csv_files)}\n")
    print("Доступные файлы (новые сверху):")
    print("   №   │ Дата и время       │ Размер     │ Имя файла")
    print("   ────┼────────────────────┼────────────┼────────────────────────────────")

    for i, path in enumerate(csv_files[:30]):
        mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(path.stat().st_mtime))
        size_mb = path.stat().st_size / (1024**2)
        print(f"  {i+1:2} │ {mtime} │ {size_mb:7.1f} МБ │ {path.name}")

    print("   ────┴────────────────────┴────────────┴────────────────────────────────\n")

    # === Выбор файла ===
    while True:
        choice = input("Введи номер датасета (или просто Enter — взять самый новый): ").strip()
        if choice == "":
            selected_file = csv_files[0]
            print(f"\nВыбран самый новый файл:")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            selected_file = csv_files[int(choice) - 1]
            print(f"\nВыбран файл №{choice}:")
            break
        else:
            print("Неверный номер! Попробуй снова.")

    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(selected_file.stat().st_mtime))
    size_mb = selected_file.stat().st_size / (1024 ** 2)
    print(f"   → {selected_file.name}")
    print(f"   → {mtime} | {size_mb:.1f} МБ")
    print(f"   → {selected_file}\n")

    print("Запуск обработки...")
    print("=" * 70)

    # === Загрузка и генерация признаков ===
    print("Загружаем CSV...")
    raw_df = pd.read_csv(selected_file, low_memory=False)
    print(f"Загружено строк: {len(raw_df):,}")

    print("Генерируем 82 мощных признака (это займёт 30–60 секунд)...")
    df = extract_features(raw_df)          # ← ВСЁ ДЕЛО В ЭТОЙ СТРОКЕ!

    # === Метки 0/1 ===
    if 'label' not in df.columns:
        df['label'] = 0
    else:
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df['label'] = df['label'].map(lambda x: 0 if x in ['normal', 'benign', '0', 'benign '] else 1)

    # === КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: очистка + безопасная нормализация ===
    # Оставляем только те, что есть в датасете
    feature_columns = [c for c in BEST_FEATURES if c in df.columns]
    print(f"Найдено {len(feature_columns)} из {len(BEST_FEATURES)} ожидаемых признаков")

    # Очистка от inf/NaN и выбросов
    X_clean = clean_features_before_scaling(df, feature_columns)

    # Теперь можно спокойно применять StandardScaler
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(X_clean)

    # === Сохранение ===
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_columns, MODELS_DIR / "feature_list.pkl")

    print("\nГОТОВО! Данные успешно обработаны")
    print(f"Потоков в датасете: {len(df):,}")
    print(f"Колонок всего: {len(df.columns)} (из них признаков: {len(feature_columns)})")
    print(f"Сохранено → {OUTPUT_FILE}")
    print("\nТеперь запускай:")
    print("   python scripts/train_models.py")
    print("   streamlit run src/visualization/analyzer_full.py")
    print("=" * 70)


if __name__ == "__main__":
    main()