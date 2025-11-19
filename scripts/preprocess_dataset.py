# scripts/preprocess_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
import yaml

# Загружаем конфиг
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATA_DIR = Path("data/raw/cic-ids-2017")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "friday_features.csv"


def main():
    print("Ищем готовые CSV от CIC-IDS2017...")

    # Вариант 1: единый файл
    unified_csv = DATA_DIR / "GeneratedLabelledFlows.csv"
    if unified_csv.exists():
        print("Найден GeneratedLabelledFlows.csv — используем его!")
        df = pd.read_csv(unified_csv)
    else:
        # Вариант 2: отдельные CSV пятницы
        csv_files = list((DATA_DIR / "CSVs").glob("*Friday*.pcap_ISCX.csv"))
        if not csv_files:
            raise FileNotFoundError("Не найдено CSV в data/raw/cic-ids-2017/CSVs/")

        print(f"Найдено {len(csv_files)} CSV-файлов — объединяем...")
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Всего строк после объединения: {len(df)}")

    # Убираем пробелы в названиях колонок
    df.columns = [c.strip() for c in df.columns]

    # --------------------- ПЕРЕИМЕНОВАНИЕ И СОЗДАНИЕ НУЖНЫХ КОЛОНОК ---------------------
    # Общие колонки, которые точно есть
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

    # Считаем общие пакеты и байты
    df['packets'] = df['fwd_packets'].fillna(0) + df['bwd_packets'].fillna(0)
    df['bytes'] = df['fwd_bytes'].fillna(0) + df['bwd_bytes'].fillna(0)

    # Длительность в секундах
    df['duration'] = df['Flow Duration'] / 1_000_000  # микросекунды → секунды
    df['duration'] = df['duration'].replace(0, 0.001)  # избегаем деления на 0

    # Если pps/bps нет — считаем вручную
    if 'pps' not in df.columns or df['pps'].isna().all():
        df['pps'] = df['packets'] / df['duration']
    if 'bps' not in df.columns or df['bps'].isna().all():
        df['bps'] = df['bytes'] / df['duration']

    # --------------------- МЕТКИ ---------------------
    label_map = {
        'BENIGN': 'normal',
        'DDoS': 'ddos',
        'PortScan': 'scan',
        'Bot': 'botnet',
        'DoS Hulk': 'dos',
        'DoS GoldenEye': 'dos',
        'DoS slowloris': 'dos',
        'DoS Slowhttptest': 'dos',
        'Heartbleed': 'heartbleed',
        'Web Attack � Brute Force': 'web_attack',
        'Web Attack � XSS': 'web_attack',
        'Web Attack � Sql Injection': 'web_attack',
        'Infiltration': 'infiltration'
    }
    df['label'] = df['Label'].map(label_map).fillna('other')

    # --------------------- ВЫБОР ФИЧЕЙ ---------------------
    feature_cols = ['pps', 'bps', 'duration', 'packets', 'bytes']

    # Добавляем дополнительные хорошие фичи, если есть
    extra_features = [
        'Avg Packet Size', 'Packet Length Mean', 'Packet Length Std',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
        'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count', 'FIN Flag Count'
    ]
    for col in extra_features:
        if col in df.columns:
            feature_cols.append(col)

    # Оставляем только нужные колонки
    final_cols = feature_cols + ['label']
    final_df = df[final_cols].copy()

    # Заменяем inf и NaN
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    final_df = final_df.fillna(0)

    # Нормализация
    scaler = StandardScaler()
    final_df[feature_cols] = scaler.fit_transform(final_df[feature_cols])

    # Сохраняем
    final_df.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, 'models/scaler.pkl')

    print("\nГОТОВО! Всё успешно обработано!")
    print(f"→ Файл: {OUTPUT_FILE}")
    print(f"→ Строк: {len(final_df)}")
    print(f"→ Метки: {final_df['label'].value_counts().to_dict()}")
    print(f"→ Фич: {len(feature_cols)} (pps, bps, duration, packets, bytes + extra)")


if __name__ == "__main__":
    main()