import pyshark
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
from src.preprocessing.flow_aggregator import aggregate_flows  # Создадим ниже
from src.preprocessing.feature_extractor import extract_features  # Создадим ниже

# Загрузи config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def main():
    raw_path = Path(config['dataset']['path'])
    output_path = Path('data/processed/flows_cic2017.csv')

    # Шаг 1: Чтение PCAP (ingestion)
    print("Чтение PCAP...")
    cap = pyshark.FileCapture(str(raw_path / config['dataset']['pcap_files'][0]))
    packets = []
    for pkt in cap:
        if 'IP' in pkt:
            packets.append({
                'timestamp': float(pkt.sniff_timestamp),
                'src_ip': pkt.ip.src,
                'dst_ip': pkt.ip.dst,
                'src_port': int(pkt[pkt.transport_layer].srcport),
                'dst_port': int(pkt[pkt.transport_layer].dstport),
                'protocol': pkt.ip.proto,
                'length': int(pkt.length)
            })
    cap.close()
    df = pd.DataFrame(packets)

    # Добавь метки (для CIC: по времени/файлу, упрощенно)
    df['label'] = 'normal'  # Замени на реальные метки из CSV-датасета
    attack_start = 10000  # Пример: с пакета 10000 - DDoS
    df.loc[df.index >= attack_start, 'label'] = 'ddos'

    # Шаг 2: Агрегация flows + фичи
    flows = aggregate_flows(df)
    features_df = extract_features(flows)

    # Нормализация
    scaler = StandardScaler()
    features_df[features_df.columns[:-2]] = scaler.fit_transform(features_df[features_df.columns[:-2]])  # Без ts/label

    # Сохрани
    features_df.to_csv(output_path, index=False)
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"Готово: {output_path}")


if __name__ == '__main__':
    main()