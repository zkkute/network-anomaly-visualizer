# scripts/preprocess_universal.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import hashlib
import warnings
warnings.filterwarnings("ignore")

# === ПУТИ ===
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
for p in [PROCESSED_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "my_traffic_features.csv"

# === ЕДИНЫЙ СПИСОК ВСЕХ ВОЗМОЖНЫХ ФИЧ (ВАЖНО!) ===
ALL_FEATURES = [
    'duration', 'packets', 'bytes', 'pps', 'bps', 'avg_pkt_size',
    'port_entropy', 'iat_std', 'unique_dst_port', 'unique_dst_ip',
    'syn_ratio', 'rst_ratio', 'fin_ratio',
    'fwd_bytes', 'bwd_bytes', 'fwd_packets', 'bwd_packets'
]


def load_and_detect_dataset(path: Path) -> pd.DataFrame:
    """Автоопределение типа датасета и загрузка"""
    print(f"Загружаем: {path.name}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Определяем тип по колонкам
    if 'Flow Duration' in df.columns or 'Total Fwd Packets' in df.columns:
        print("→ Обнаружен CIC-IDS2017 / CSE-CIC-IDS2018")
        return df, "cic"
    elif 'dur' in df.columns and 'proto' in df.columns and 'sbytes' in df.columns:
        print("→ Обнаружен UNSW-NB15")
        return df, "unsw"
    elif 'Timestamp' in df.columns and 'Src IP' in df.columns:
        print("→ Обнаружен CAIDA / личный CSV из Wireshark")
        return df, "custom"
    else:
        print("→ Неизвестный формат, попробуем как кастомный")
        return df, "custom"


def extract_advanced_features(df_flows: pd.DataFrame) -> pd.DataFrame:
    """Добавляем продвинутые фичи на уровне потоков или окон"""
    df = df_flows.copy()

    # Базовые фичи
    df['duration'] = df['duration'].replace(0, 0.001)
    df['packets'] = df['packets'].fillna(0).astype(int)
    df['bytes'] = df['bytes'].fillna(0).astype(int)
    df['pps'] = df['packets'] / df['duration']
    df['bps'] = df['bytes'] / df['duration']
    df['avg_pkt_size'] = df['bytes'] / df['packets'].replace(0, 1)

    # Флаги (если есть)
    for flag in ['SYN', 'RST', 'FIN', 'ACK', 'PSH', 'URG']:
        col = f"{flag.lower()}_ratio"
        count_col = f"{flag} Flag Count" if f"{flag} Flag Count" in df else f"{flag} Flag Cnt"
        if count_col in df.columns:
            df[col] = df[count_col] / df['packets'].replace(0, 1)
        else:
            df[col] = 0.0

    df['syn_ratio'] = df.get('syn_ratio', df.get('SYN Flag Count', 0)) / df['packets'].replace(0, 1)
    df['rst_ratio'] = df.get('rst_ratio', 0)
    df['fin_ratio'] = df.get('fin_ratio', 0)

    # Энтропия портов и уникальные порты (если есть dst_port)
    if 'dst_port' in df.columns:
        port_counts = df['dst_port'].value_counts(normalize=True)
        df['port_entropy'] = df['dst_port'].map(port_counts).fillna(0)
        df['port_entropy'] = df['port_entropy'].apply(lambda x: entropy([x, 1-x]) if x > 0 else 0)
        df['unique_dst_port'] = df['dst_port'].nunique()
    else:
        df['port_entropy'] = 0.0
        df['unique_dst_port'] = 1

    # IAT std (если есть временные метки пакетов — упрощённо)
    df['iat_std'] = 0.0

    # Направленность
    for col in ['fwd_bytes', 'bwd_bytes', 'fwd_packets', 'bwd_packets']:
        df[col] = df.get(col, 0)

    df['unique_dst_ip'] = 1  # заглушка, можно улучшить при агрегации

    return df


def process_cic_ids2017(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={
        'Flow Duration': 'duration',
        'Total Fwd Packets': 'fwd_packets',
        'Total Backward Packets': 'bwd_packets',
        'Total Length of Fwd Packets': 'fwd_bytes',
        'Total Length of Bwd Packets': 'bwd_bytes',
        'Flow Packets/s': 'pps',
        'Flow Bytes/s': 'bps',
        'Label': 'label_raw'
    }, inplace=True)

    df['duration'] = df['duration'] / 1_000_000
    df['packets'] = df['fwd_packets'].fillna(0) + df['bwd_packets'].fillna(0)
    df['bytes'] = df['fwd_bytes'].fillna(0) + df['bwd_bytes'].fillna(0)

    # Метки
    benign = ['BENIGN', 'Benign']
    attack_map = {
        'DDoS': 'ddos', 'PortScan': 'scan', 'Bot': 'botnet',
        'DoS Hulk': 'dos', 'DoS GoldenEye': 'dos', 'DoS slowloris': 'dos',
        'Heartbleed': 'heartbleed', 'Web Attack': 'web_attack',
        'Infiltration': 'infiltration', 'FTP-Patator': 'bruteforce', 'SSH-Patator': 'bruteforce'
    }
    df['label'] = df['label_raw'].apply(lambda x: 'normal' if str(x) in benign else attack_map.get(str(x), 'other'))

    df = extract_advanced_features(df)
    return df


def process_unsw_nb15(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={
        'dur': 'duration',
        'spkts': 'fwd_packets',
        'dpkts': 'bwd_packets',
        'sbytes': 'fwd_bytes',
        'dbytes': 'bwd_bytes',
        'rate': 'pps',
        'srate': 'pps_fwd',
        'Label': 'label'
    }, inplace=True)

    df['packets'] = df['fwd_packets'] + df['bwd_packets']
    df['bytes'] = df['fwd_bytes'] + df['bwd_bytes']
    df['bps'] = df['bytes'] / df['duration'].replace(0, 0.001)
    df['label'] = df['label'].apply(lambda x: 'normal' if x == 0 else 'attack')

    df = extract_advanced_features(df)
    return df


def process_custom(df: pd.DataFrame) -> pd.DataFrame:
    # Для твоих Wireshark CSV
    required = ['Time', 'Source', 'Destination', 'Length', 'Info']
    if not all(c in df.columns for c in required):
        raise ValueError("Не хватает колонок: Time, Source, Destination, Length, Info")

    df['timestamp'] = pd.to_numeric(df['Time'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()

    # Парсинг портов из Info (очень надёжно)
    df['src_port'] = df['Info'].str.extract(r'(\d+) +>')[0]
    df['dst_port'] = df['Info'].str.extract(r'> +(\d+)')[0]
    df['src_port'] = pd.to_numeric(df['src_port'], errors='coerce').fillna(0).astype(int)
    df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce').fillna(443).astype(int)

    # Группировка в потоки
    df['flow_key'] = (
        df['Source'] + "_" + df['src_port'].astype(str) + "_" +
        df['Destination'] + "_" + df['dst_port'].astype(str)
    )

    flows = []
    for key, group in df.groupby('flow_key'):
        if len(group) < 2: continue
        duration = group['timestamp'].max() - group['timestamp'].min()
        duration = max(duration, 0.001)
        flows.append({
            'duration': duration,
            'packets': len(group),
            'bytes': group['Length'].sum(),
            'pps': len(group) / duration,
            'bps': group['Length'].sum() / duration,
            'avg_pkt_size': group['Length'].mean(),
            'dst_port': group['dst_port'].iloc[0],
            'label': 'normal'
        })

    result = pd.DataFrame(flows)
    result = extract_advanced_features(result)
    return result


def main():
    print("УНИВЕРСАЛЬНЫЙ ПРЕПРОЦЕССОР v2.0")
    print("=" * 70)

    files = []
    # Готовые обработанные
    ready_files = list(PROCESSED_DIR.glob("*.csv"))
    for f in ready_files:
        if f.name != OUTPUT_FILE.name:
            files.append(("R" + str(len(files)+1), f"Готовый: {f.name}", f, "ready"))

    # Сырые
    for subdir in ["cic-ids-2017/CSVs", "unsw_nb15", "caida", ""]:
        path = RAW_DIR / subdir
        if path.exists():
            for csv in path.glob("*.csv"):
                files.append((str(len(files)+1), f"{subdir or 'custom'}: {csv.name}", csv, "raw"))

    if not files:
        print("Нет файлов! Положи в data/raw/")
        return

    print("Доступно:")
    for num, text, _, _ in files[:20]:
        print(f"  [{num}] {text}")
    if len(files) > 20:
        print(f"  ... и ещё {len(files)-20} файлов")

    choice = input("\nВыбери номер → ").strip()
    selected = next((f for f in files if f[0] == choice), None)
    if not selected:
        print("Ничего не выбрано!")
        return

    _, desc, path, typ = selected
    print(f"\nОбрабатываем: {desc}")

    if typ == "ready":
        df = pd.read_csv(path)
    else:
        raw_df, dataset_type = load_and_detect_dataset(path)
        if dataset_type == "cic":
            df = process_cic_ids2017(raw_df)
        elif dataset_type == "unsw":
            df = process_unsw_nb15(raw_df)
        else:
            df = process_custom(raw_df)

    # Оставляем только нужные фичи
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing_features = [f for f in ALL_FEATURES if f not in df.columns]
    for f in missing_features:
        df[f] = 0.0

    df = df[ALL_FEATURES + ['label']]

    # Нормализация
    scaler = StandardScaler()
    df[ALL_FEATURES] = scaler.fit_transform(df[ALL_FEATURES])

    # Сохраняем
    df.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(ALL_FEATURES, MODELS_DIR / "feature_list.pkl")

    print("\nГОТОВО!")
    print(f"Потоков: {len(df):,}")
    print(f"Фич: {len(ALL_FEATURES)} (добавлены: port_entropy, iat_std и др.)")
    print(f"Файл → {OUTPUT_FILE}")
    print(f"Scaler и feature_list → models/")
    print("\nТеперь запускай:")
    print("   python scripts/train_models.py")
    print("   streamlit run src/visualization/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()