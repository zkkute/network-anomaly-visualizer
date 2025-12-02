# src/preprocessing/feature_extractor.py
import pandas as pd
import numpy as np
from scipy.stats import entropy
from typing import List, Optional, Dict
import warnings

warnings.filterwarnings("ignore")

# === ГЛОБАЛЬНЫЙ СПИСОК ФИЧ — ДОЛЖЕН СОВПАДАТЬ С preprocess_universal_v2.py ===
ALL_FEATURES = [
    'duration', 'packets', 'bytes', 'pps', 'bps', 'avg_pkt_size',
    'port_entropy', 'iat_std', 'unique_dst_port', 'unique_dst_ip',
    'syn_ratio', 'rst_ratio', 'fin_ratio',
    'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes'
]


def create_flow_key(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт надёжный flow_key: src_ip:port → dst_ip:port_protocol"""
    df = df.copy()

    required = ['src_ip', 'dst_ip']
    for col in required:
        if col not in df.columns:
            df[col] = '192.168.1.1'  # fallback

    if 'src_port' not in df.columns:
        df['src_port'] = 0
    if 'dst_port' not in df.columns:
        df['dst_port'] = np.random.choice([80, 443, 22], size=len(df))
    if 'protocol' not in df.columns:
        df['protocol'] = 'TCP'

    df['src_port'] = pd.to_numeric(df['src_port'], errors='coerce').fillna(0).astype(int)
    df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce').fillna(443).astype(int)

    df['flow_key'] = (
            df['src_ip'].astype(str) + "_" +
            df['src_port'].astype(str) + "→" +
            df['dst_ip'].astype(str) + "_" +
            df['dst_port'].astype(str) + "_" +
            df['protocol'].astype(str).str.upper()
    )
    return df


def extract_flow_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Основная функция: принимает сырой DataFrame (из CSV или PCAP),
    возвращает датафрейм с фичами на уровне потоков + label
    """
    df = df.copy()

    # === 1. Базовая нормализация колонок ===
    rename_map = {
        'Flow Duration': 'duration_raw', 'dur': 'duration_raw', 'Duration': 'duration_raw',
        'Tot Fwd Pkts': 'fwd_packets', 'Fwd Pkts': 'fwd_packets', 'spkts': 'fwd_packets',
        'Tot Bwd Pkts': 'bwd_packets', 'Bwd Pkts': 'bwd_packets', 'dpkts': 'bwd_packets',
        'TotLen Fwd Pkts': 'fwd_bytes', 'Fwd Bytes': 'fwd_bytes', 'sbytes': 'fwd_bytes',
        'TotLen Bwd Pkts': 'bwd_bytes', 'Bwd Bytes': 'bwd_bytes', 'dbytes': 'bwd_bytes',
        'SYN Flag Cnt': 'syn_count', 'FIN Flag Cnt': 'fin_count', 'RST Flag Cnt': 'rst_count',
        'Label': 'label_raw', 'attack': 'label_raw'
    }
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns=rename_map, inplace=True)

    # === 2. Создание flow_key и группировка ===
    df = create_flow_key(df)

    if 'timestamp' not in df.columns:
        df['timestamp'] = np.arange(len(df))

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # === 3. Агрегация по потокам ===
    flow_groups = []
    for flow_key, group in df.groupby('flow_key', group_keys=False):
        if len(group) == 0:
            continue

        duration = group['timestamp'].max() - group['timestamp'].min()
        duration = max(duration, 0.001)

        packets = len(group)
        bytes_total = group['Length'].sum() if 'Length' in group.columns else 100 * packets

        fwd_packets = group['fwd_packets'].sum() if 'fwd_packets' in group.columns else packets // 2
        bwd_packets = group['bwd_packets'].sum() if 'bwd_packets' in group.columns else packets - fwd_packets
        fwd_bytes = group['fwd_bytes'].sum() if 'fwd_bytes' in group.columns else bytes_total // 2
        bwd_bytes = group['bwd_bytes'].sum() if 'bwd_bytes' in group.columns else bytes_total - fwd_bytes

        syn_count = group['syn_count'].sum() if 'syn_count' in group.columns else 0
        rst_count = group['rst_count'].sum() if 'rst_count' in group.columns else 0
        fin_count = group['fin_count'].sum() if 'fin_count' in group.columns else 0

        iat = group['timestamp'].diff().dropna()
        iat_std = iat.std() if len(iat) > 1 else 0.0

        row = {
            'flow_key': flow_key,
            'duration': duration,
            'packets': packets,
            'bytes': bytes_total,
            'fwd_packets': fwd_packets,
            'bwd_packets': bwd_packets,
            'fwd_bytes': fwd_bytes,
            'bwd_bytes': bwd_bytes,
            'pps': packets / duration,
            'bps': bytes_total / duration,
            'avg_pkt_size': bytes_total / packets if packets > 0 else 0,
            'syn_ratio': syn_count / packets if packets > 0 else 0,
            'rst_ratio': rst_count / packets if packets > 0 else 0,
            'fin_ratio': fin_count / packets if packets > 0 else 0,
            'iat_std': iat_std,
            'unique_dst_port': group['dst_port'].nunique(),
            'unique_dst_ip': group['dst_ip'].nunique(),
            'port_entropy': entropy(group['dst_port'].value_counts(normalize=True)) if group[
                                                                                           'dst_port'].nunique() > 1 else 0.0,
            'timestamp_start': group['timestamp'].min(),
            'timestamp_end': group['timestamp'].max(),
        }
        flow_groups.append(row)

    result = pd.DataFrame(flow_groups)

    # === 4. Метки ===
    if 'label_raw' in df.columns:
        # Берём метку из первого пакета потока
        label_map = df[['flow_key', 'label_raw']].drop_duplicates('flow_key').set_index('flow_key')['label_raw']
        result = result.merge(label_map, left_on='flow_key', right_index=True, how='left')
        result['label'] = result['label_raw'].fillna('normal')
        result['label'] = result['label'].astype(str).str.lower()
        result['label'] = result['label'].replace({
            'benign': 'normal', '0': 'normal', 'normal': 'normal',
            'ddos': 'ddos', 'portscan': 'scan', 'bot': 'botnet', '1': 'attack'
        }).fillna('other')
    else:
        result['label'] = 'normal'

    # === 5. Финальная очистка ===
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Оставляем только нужные колонки
    final_cols = [f for f in ALL_FEATURES if f in result.columns] + ['label', 'timestamp_start', 'timestamp_end']
    missing = [f for f in ALL_FEATURES if f not in result.columns]
    for f in missing:
        result[f] = 0.0

    return result[final_cols]


def get_feature_list() -> List[str]:
    """Возвращает точный список фич — используется в обучении и inference"""
    return ALL_FEATURES.copy()


def get_feature_importance_info() -> Dict[str, str]:
    """Для диплома — описание, что означает каждая фича"""
    return {
        'pps': 'Пакетов в секунду — главный признак DDoS',
        'bps': 'Байт в секунду — volumetric атаки и утечки',
        'port_entropy': 'Разнообразие портов — сканирование',
        'iat_std': 'Нестабильность межпакетных интервалов — C2, ботнеты',
        'unique_dst_port': 'Сколько разных портов трогает один IP — сканирование',
        'syn_ratio': 'Доля SYN-пакетов — SYN-flood, сканирование',
        'avg_pkt_size': 'Средний размер пакета — малый = DoS, большой = exfiltration',
    }

extract_features = extract_flow_level_features