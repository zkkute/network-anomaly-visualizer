def extract_features(flows):
    """20 фич"""
    features = flows[['pps', 'bps', 'duration', 'packets', 'bytes']].copy()
    features['syn_ratio'] = 0.5  # Упрощенно; в реале из флагов
    features['port_entropy'] = np.log(features['src_port'].nunique() + 1)  # Пример
    features['label'] = flows['label']  # Метка
    features['ts_start'] = flows['ts_start']
    return features.sort_values('ts_start')