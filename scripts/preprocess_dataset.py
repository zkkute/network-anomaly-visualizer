
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "my_traffic_features.csv"


def process_cic_ids2017(path):
    """Точно как твой старый preprocess_dataset.py — проверено на всех файлах CIC-IDS2017"""
    print(f"Обрабатываем CIC-IDS2017 датасет: {path.name}")
    df = pd.read_csv(path)

    # Убираем лишние пробелы в названиях колонок
    df.columns = [c.strip() for c in df.columns]

    # Переименовываем нужные колонки
    df.rename(columns={
        'Flow Duration': 'Flow Duration',
        'Total Fwd Packets': 'fwd_packets',
        'Total Backward Packets': 'bwd_packets',
        'Total Length of Fwd Packets': 'fwd_bytes',
        'Total Length of Bwd Packets': 'bwd_bytes',
        'Fwd Packets/s': 'pps_fwd',
        'Bwd Packets/s': 'pps_bwd',
        'Flow Packets/s': 'pps',
        'Flow Bytes/s': 'bps',
        'Label': 'Label'
    }, inplace=True)

    # Общие значения
    df['packets'] = df['fwd_packets'].fillna(0) + df['bwd_packets'].fillna(0)
    df['bytes'] = df['fwd_bytes'].fillna(0) + df['bwd_bytes'].fillna(0)
    df['duration'] = df['Flow Duration'] / 1_000_000
    df['duration'] = df['duration'].replace(0, 0.001)

    # pps и bps, если их нет
    if 'pps' not in df.columns or df['pps'].isna().all():
        df['pps'] = df['packets'] / df['duration']
    if 'bps' not in df.columns or df['bps'].isna().all():
        df['bps'] = df['bytes'] / df['duration']

    # Метки — как у тебя было
    label_map = {
        'BENIGN': 'normal', 'Benign': 'normal',
        'DDoS': 'ddos', 'DDOS': 'ddos',
        'PortScan': 'scan',
        'Bot': 'botnet',
        'DoS Hulk': 'dos', 'DoS GoldenEye': 'dos',
        'DoS slowloris': 'dos', 'DoS Slowhttptest': 'dos',
        'Heartbleed': 'heartbleed',
        'Web Attack - Brute Force': 'web_attack',
        'Web Attack - XSS': 'web_attack',
        'Web Attack - Sql Injection': 'web_attack',
        'Infiltration': 'infiltration',
        'FTP-Patator': 'bruteforce', 'SSH-Patator': 'bruteforce'
    }
    df['label'] = df['Label'].map(label_map).fillna('other')

    # Базовые фичи
    feature_cols = ['pps', 'bps', 'duration', 'packets', 'bytes']

    # Дополнительные (если есть)
    extra = [
        'Avg Packet Size', 'Packet Length Mean', 'Packet Length Std',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
        'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count', 'FIN Flag Count',
        'PSH Flag Count', 'URG Flag Count'
    ]
    for col in extra:
        if col in df.columns:
            feature_cols.append(col)

    final_df = df[feature_cols + ['label']].copy()
    final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    final_df[feature_cols] = scaler.fit_transform(final_df[feature_cols])

    return final_df, scaler, len(final_df)


def process_wireshark(path):
    """Твой личный захват из Wireshark"""
    print(f"Обрабатываем твой реальный трафик: {path.name}")
    # (Тот же код, что был раньше — работает отлично)
    df = pd.read_csv(path)
    df = df.rename(columns={
        "No.": "no", "Time": "timestamp", "Source": "src_ip",
        "Destination": "dst_ip", "Protocol": "protocol",
        "Length": "length", "Info": "info"
    })

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()

    df['src_port'] = df['info'].str.extract(r'(\d+)\s+>\s+\d+')[0]
    df['dst_port'] = df['info'].str.extract(r'>\s+(\d+)')[0]
    df['src_port'] = df['src_port'].fillna(df['info'].str.split(' > ').str[0].str.split().str[-1])
    df['dst_port'] = df['dst_port'].fillna(df['info'].str.split(' > ').str[1].str.split().str[0])
    df['dst_port'] = df['dst_port'].fillna(df['info'].apply(lambda x: 443 if '443' in str(x) else 80))
    df['src_port'] = df['src_port'].fillna(0)

    df['src_port'] = pd.to_numeric(df['src_port'], errors='coerce').fillna(0).astype(int)
    df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce').fillna(0).astype(int)
    df['protocol'] = df['protocol'].fillna('TCP').str.upper()

    df['flow_key'] = (
        df['src_ip'] + ":" + df['src_port'].astype(str) + "→" +
        df['dst_ip'] + ":" + df['dst_port'].astype(str) + "_" +
        df['protocol']
    )

    flows = []
    for _, group in df.groupby('flow_key'):
        if len(group) < 2: continue
        duration = max(group['timestamp'].max() - group['timestamp'].min(), 0.001)
        packets = len(group)
        bytes_total = group['length'].sum()
        flows.append({
            'duration': duration, 'packets': packets, 'bytes': bytes_total,
            'pps': packets / duration, 'bps': bytes_total / duration,
            'avg_pkt_size': bytes_total / packets if packets > 0 else 0,
            'label': 'normal'
        })

    final_df = pd.DataFrame(flows)
    if final_df.empty:
        raise ValueError("Не удалось создать ни одного потока из твоего файла")

    feature_cols = ['duration', 'packets', 'bytes', 'pps', 'bps', 'avg_pkt_size']
    final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    final_df[feature_cols] = scaler.fit_transform(final_df[feature_cols])

    return final_df, scaler, len(final_df)


def main():
    print("=" * 70)
    print("   УНИВЕРСАЛЬНЫЙ ПРЕПРОЦЕССОР — ВЫБЕРИ, ЧТО ОБРАБОТАТЬ")
    print("=" * 70)

    options = []
    my_file = RAW_DIR / "test.csv"
    if my_file.exists():
        options.append(("M", "Мой реальный захват (test.csv)", my_file, "wireshark"))

    cic_files = list((RAW_DIR / "cic-ids-2017" / "CSVs").glob("*.csv"))
    for i, f in enumerate(cic_files, 1):
        name = f.name
        if "DDoS" in name: name += " (DDoS!)"
        if "PortScan" in name: name += " (сканирование)"
        options.append((str(i), f"CIC-IDS2017: {name}", f, "cic"))

    if (PROCESSED_DIR / "friday_features.csv").exists():
        options.append(("R", "Быстро: использовать готовый friday_features.csv", None, "ready"))

    if not options:
        print("Ничего не найдено! Положи файлы в data/raw/ или data/raw/cic-ids-2017/CSVs/")
        return

    print("\nДоступно:")
    for num, text, _, _ in options:
        print(f"   [{num}] {text}")
    print("   [Q] Выход")

    choice = input("\nВыбери → ").strip().upper()
    if choice == "Q": return

    selected = next((opt for opt in options if opt[0] == choice), None)
    if not selected:
        print("Неправильно выбрано!")
        return

    _, desc, path, typ = selected

    if typ == "ready":
        print("Копируем готовый датасет...")
        df = pd.read_csv(PROCESSED_DIR / "friday_features.csv")
        df.to_csv(OUTPUT_FILE, index=False)
        print("Готово!")
        return

    print(f"\n{desc}")

    if typ == "cic":
        final_df, scaler, count = process_cic_ids2017(path)
    else:  # wireshark
        final_df, scaler, count = process_wireshark(path)

    final_df.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    print("\nГОТОВО!")
    print(f"   Потоков обработано: {count}")
    print(f"   Файл: {OUTPUT_FILE}")
    print(f"   Scaler: models/scaler.pkl")
    print("\nТеперь запускай:")
    print("   python scripts/train_models.py")
    print("   streamlit run src/visualization/demo_heatmap.py")
    print("=" * 70)


if __name__ == "__main__":
    main()