import pandas as pd
import numpy as np
from scipy.stats import entropy
import warnings
import re
warnings.filterwarnings("ignore")

# === ВСЕ признаки: старые + 27 новых ===
ALL_FEATURES = [
    # оригинальные
    'duration', 'packets', 'bytes', 'pps', 'bps', 'avg_pkt_size',
    'port_entropy', 'iat_std', 'unique_dst_port', 'unique_dst_ip',
    'syn_ratio', 'rst_ratio', 'fin_ratio',
    'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',

    # +27 новых (IAT, pkt len, ratios, flags, entropy, burst, throughput)
    # IAT statistics
    'iat_mean', 'iat_max', 'iat_min', 'iat_var', 'iat_skew', 'iat_kurtosis',

    # packet length statistics
    'pkt_len_mean', 'pkt_len_max', 'pkt_len_min', 'pkt_len_std', 'pkt_len_var', 'pkt_len_skew', 'pkt_len_kurtosis',

    # direction / ratios
    'fwd_bwd_ratio_packets', 'fwd_bwd_ratio_bytes', 'bytes_per_pkt', 'pkts_per_byte',

    # entropy
    'ip_dst_entropy', 'port_dst_entropy',

    # tcp flags (ratios)
    'ack_ratio', 'psh_ratio', 'urg_ratio', 'ece_ratio', 'cwr_ratio',

    # burst / activity
    'flow_active_time', 'flow_idle_time',

    # extra throughput proxies
    'bytes_per_second', 'packets_per_second'
]


def _col_find(cols, *patterns):
    cols_low = [c.lower() for c in cols]
    for i, c in enumerate(cols_low):
        if all(p.lower() in c for p in patterns):
            return cols[i]
    return None


def _any_col_contains(cols, pattern):
    pat = pattern.lower()
    for c in cols:
        if pat in c.lower():
            return c
    return None


def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)


def create_flow_key(df):
    df = df.copy()
    # Безопасно создаём колонки
    for col in ['src_ip', 'dst_ip']:
        if col not in df.columns:
            df[col] = '0.0.0.0'
    for col in ['src_port', 'dst_port']:
        if col not in df.columns:
            df[col] = 0
    if 'protocol' not in df.columns:
        df['protocol'] = 'TCP'

    df['src_port'] = pd.to_numeric(df['src_port'], errors='coerce').fillna(0).astype(int)
    df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce').fillna(0).astype(int)

    df['flow_key'] = (
        df['src_ip'].astype(str) + ':' + df['src_port'].astype(str) + ' → ' +
        df['dst_ip'].astype(str) + ':' + df['dst_port'].astype(str) + ' ' +
        df['protocol'].astype(str)
    )
    return df


def _is_flow_level(df):
    cols = [c.strip() for c in df.columns]
    # заметные признаки flow-level
    if any(re.search(r'flow.*duration', c, re.I) for c in cols) or any(c.lower().strip() == 'label' for c in cols):
        return True
    # fallback: если нет src_ip/ dst_ip но много числовых колонок
    if not any('src_ip' in c.lower() or 'dst_ip' in c.lower() for c in cols) and \
            sum(np.issubdtype(dtype, np.number) for dtype in df.dtypes) > 5:
        return True
    return False


def _series_stats(s: pd.Series):
    """Возвращает mean, max, min, var, skew, kurtosis — безопасно."""
    if s is None or len(s) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    s = s.dropna()
    if len(s) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return float(s.mean()), float(s.max()), float(s.min()), float(s.var(ddof=0)), float(s.skew()) if hasattr(s, 'skew') else 0.0, float(s.kurt()) if hasattr(s, 'kurt') else 0.0


def extract_flow_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Универсальный extractor.
    Если на входе уже flow-level — используем имеющиеся колонки (с безопасными прокси).
    Если на входе packet-level — группируем по flow_key и рассчитываем все признаки.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # нормализованный map
    col_map = {
        'Flow Duration': 'Flow_Duration', 'TotLen Fwd Pkts': 'TotLen_Fwd_Pkts',
        'TotLen Bwd Pkts': 'TotLen_Bwd_Pkts', 'Packet Length Mean': 'Pkt_Len_Mean',
        'Src IP': 'src_ip', 'Dst IP': 'dst_ip', 'Src Port': 'src_port', 'Dst Port': 'dst_port',
        'Protocol': 'protocol', 'Timestamp': 'timestamp', 'Label': 'label',
        'Fwd Packets/s': 'Fwd_Packets_per_s', 'Bwd Packets/s': 'Bwd_Packets_per_s',
        'Flow Packets/s': 'Flow_Packets_per_s', 'Flow Bytes/s': 'Flow_Bytes_per_s',
        ' Flow IAT Std': 'Flow_IAT_Std'
    }
    rename = {orig: col_map[orig] for orig in col_map if orig in df.columns}
    if rename:
        df.rename(columns=rename, inplace=True)

    # заменяем пробелы на _
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # ---------------- flow-level branch ----------------
    if _is_flow_level(df):
        cols = list(df.columns)

        # --- базовые колонки / кандидаты ---
        duration_col = _col_find(cols, 'flow', 'duration') or _any_col_contains(cols, 'flow_duration') or _any_col_contains(cols, 'flow_duration')
        fwd_pk_col = _any_col_contains(cols, 'fwd_packet') or _any_col_contains(cols, 'fwd_packets') or _any_col_contains(cols, 'subflow_fwd_packets')
        bwd_pk_col = _any_col_contains(cols, 'bwd_packet') or _any_col_contains(cols, 'bwd_packets') or _any_col_contains(cols, 'subflow_bwd_packets')
        tot_fwd_len = _any_col_contains(cols, 'total_length_of_fwd') or _any_col_contains(cols, 'totlen_fwd') or _any_col_contains(cols, 'totlen_fwd_packets') or _any_col_contains(cols, 'fwd_bytes')
        tot_bwd_len = _any_col_contains(cols, 'total_length_of_bwd') or _any_col_contains(cols, 'totlen_bwd') or _any_col_contains(cols, 'bwd_bytes')
        flow_bytes_per_s = _any_col_contains(cols, 'flow_bytes') or _any_col_contains(cols, 'flow_bytes_s') or _any_col_contains(cols, 'flow_bytes_per_s') or _any_col_contains(cols, 'flow_bytes/s')
        flow_pk_per_s = _any_col_contains(cols, 'flow_packets') or _any_col_contains(cols, 'flow_packets_s') or _any_col_contains(cols, 'flow_packets_per_s') or _any_col_contains(cols, 'flow_packets/s')

        avg_pkt_size_col = _any_col_contains(cols, 'avg_pkt') or _any_col_contains(cols, 'average_packet') or _any_col_contains(cols, 'pkt_len_mean') or _any_col_contains(cols, 'pkt_len')

        syn_col = _any_col_contains(cols, 'syn_flag') or _any_col_contains(cols, 'syn_count') or _any_col_contains(cols, 'syn')
        rst_col = _any_col_contains(cols, 'rst_flag') or _any_col_contains(cols, 'rst_count') or _any_col_contains(cols, 'rst')
        fin_col = _any_col_contains(cols, 'fin_flag') or _any_col_contains(cols, 'fin_count') or _any_col_contains(cols, 'fin')

        # новые фичи: iat / pkt_len cols
        iat_mean_col = _any_col_contains(cols, 'iat_mean') or _any_col_contains(cols, 'flow_iat_mean') or _any_col_contains(cols, 'flow_iat')
        iat_std_col = _any_col_contains(cols, 'iat_std') or _any_col_contains(cols, 'flow_iat_std')
        pkt_mean_col = _any_col_contains(cols, 'pkt_len_mean') or _any_col_contains(cols, 'pkt_len') or _any_col_contains(cols, 'pkt_len_mean')
        pkt_max_col = _any_col_contains(cols, 'pkt_len_max') or _any_col_contains(cols, 'pkt_len_max')
        pkt_min_col = _any_col_contains(cols, 'pkt_len_min')
        pkt_std_col = _any_col_contains(cols, 'pkt_len_std')

        # flag additional
        ack_col = _any_col_contains(cols, 'ack_flag') or _any_col_contains(cols, 'ack_count') or _any_col_contains(cols, 'ack')
        psh_col = _any_col_contains(cols, 'psh_flag') or _any_col_contains(cols, 'psh_count') or _any_col_contains(cols, 'psh')
        urg_col = _any_col_contains(cols, 'urg_flag') or _any_col_contains(cols, 'urg_count') or _any_col_contains(cols, 'urg')
        ece_col = _any_col_contains(cols, 'ece_flag') or _any_col_contains(cols, 'ece_count') or _any_col_contains(cols, 'ece')
        cwr_col = _any_col_contains(cols, 'cwr_flag') or _any_col_contains(cols, 'cwr_count') or _any_col_contains(cols, 'cwr')

        # activity / burst
        active_col = _any_col_contains(cols, 'active') or _any_col_contains(cols, 'flow_active')
        idle_col = _any_col_contains(cols, 'idle') or _any_col_contains(cols, 'flow_idle')

        # dst ip/port entropy if present
        ip_dst_entropy_col = _any_col_contains(cols, 'dst_ip_entropy') or _any_col_contains(cols, 'ip_dst_entropy') or _any_col_contains(cols, 'ip_dst_entropy')
        port_dst_entropy_col = _any_col_contains(cols, 'port_entropy') or _any_col_contains(cols, 'dst_port_entropy') or _any_col_contains(cols, 'port_dst_entropy')

        # prepare output frame
        out = pd.DataFrame(index=df.index)

        # --- базовые ---
        if duration_col and duration_col in df.columns:
            out['duration'] = pd.to_numeric(df[duration_col], errors='coerce').fillna(0).astype(float)
        else:
            out['duration'] = 1.0

        # packets
        if fwd_pk_col and fwd_pk_col in df.columns and bwd_pk_col and bwd_pk_col in df.columns:
            out['packets'] = pd.to_numeric(df[fwd_pk_col], errors='coerce').fillna(0) + pd.to_numeric(df[bwd_pk_col], errors='coerce').fillna(0)
        else:
            cand_fwd = _any_col_contains(cols, 'fwd_packets')
            cand_bwd = _any_col_contains(cols, 'bwd_packets')
            if cand_fwd and cand_bwd:
                out['packets'] = pd.to_numeric(df[cand_fwd], errors='coerce').fillna(0) + pd.to_numeric(df[cand_bwd], errors='coerce').fillna(0)
            else:
                out['packets'] = 1.0

        # bytes / fwd / bwd
        if tot_fwd_len and tot_fwd_len in df.columns:
            fbytes = pd.to_numeric(df[tot_fwd_len], errors='coerce').fillna(0)
        else:
            fbytes = pd.Series(0, index=df.index)
        if tot_bwd_len and tot_bwd_len in df.columns:
            bbytes = pd.to_numeric(df[tot_bwd_len], errors='coerce').fillna(0)
        else:
            bbytes = pd.Series(0, index=df.index)

        if (fbytes.sum() + bbytes.sum()) == 0 and flow_bytes_per_s and flow_bytes_per_s in df.columns:
            bytes_proxy = pd.to_numeric(df[flow_bytes_per_s], errors='coerce').fillna(0) * out['duration']
            out['bytes'] = bytes_proxy
            out['fwd_bytes'] = (bytes_proxy / 2).astype(float)
            out['bwd_bytes'] = (bytes_proxy / 2).astype(float)
        else:
            out['fwd_bytes'] = fbytes.astype(float)
            out['bwd_bytes'] = bbytes.astype(float)
            out['bytes'] = out['fwd_bytes'] + out['bwd_bytes']

        # pps / bps
        if flow_pk_per_s and flow_pk_per_s in df.columns:
            out['pps'] = pd.to_numeric(df[flow_pk_per_s], errors='coerce').fillna(0).astype(float)
        else:
            out['pps'] = out['packets'] / out['duration'].replace(0, 1)

        if flow_bytes_per_s and flow_bytes_per_s in df.columns:
            out['bps'] = pd.to_numeric(df[flow_bytes_per_s], errors='coerce').fillna(0).astype(float)
        else:
            out['bps'] = out['bytes'] / out['duration'].replace(0, 1)

        # avg_pkt_size
        if avg_pkt_size_col and avg_pkt_size_col in df.columns:
            out['avg_pkt_size'] = pd.to_numeric(df[avg_pkt_size_col], errors='coerce').fillna(0).astype(float)
        else:
            out['avg_pkt_size'] = (out['bytes'] / out['packets'].replace(0, 1)).astype(float)

        # port_entropy fallback
        out['port_entropy'] = pd.to_numeric(df.get(port_dst_entropy_col, 0), errors='coerce').fillna(0.0) if port_dst_entropy_col else 0.0

        # iat_std
        out['iat_std'] = pd.to_numeric(df.get(iat_std_col, 0), errors='coerce').fillna(0.0) if iat_std_col else 0.0

        # unique ports / ips
        out['unique_dst_port'] = 1
        out['unique_dst_ip'] = 1

        # flag ratios
        packets_num = out['packets'].replace(0, 1)
        out['syn_ratio'] = pd.to_numeric(df[syn_col], errors='coerce').fillna(0) / packets_num if syn_col else 0.0
        out['rst_ratio'] = pd.to_numeric(df[rst_col], errors='coerce').fillna(0) / packets_num if rst_col else 0.0
        out['fin_ratio'] = pd.to_numeric(df[fin_col], errors='coerce').fillna(0) / packets_num if fin_col else 0.0

        # fwd/bwd packets
        out['fwd_packets'] = pd.to_numeric(df[fwd_pk_col], errors='coerce').fillna(0).astype(int) if fwd_pk_col else (out['packets'] // 2).astype(int)
        out['bwd_packets'] = pd.to_numeric(df[bwd_pk_col], errors='coerce').fillna(0).astype(int) if bwd_pk_col else (out['packets'] - out['fwd_packets']).astype(int)

        # === НОВЫЕ признаки (flow-level) ===
        # IAT stats: если есть готовые колонки — брать, иначе 0
        out['iat_mean'] = pd.to_numeric(df.get(iat_mean_col, 0), errors='coerce').fillna(0) if iat_mean_col else 0.0
        out['iat_max'] = pd.to_numeric(df.get(_any_col_contains(cols, 'iat_max') if _any_col_contains(cols, 'iat_max') else None, 0), errors='coerce').fillna(0) if _any_col_contains(cols, 'iat_max') else 0.0
        out['iat_min'] = pd.to_numeric(df.get(_any_col_contains(cols, 'iat_min') if _any_col_contains(cols, 'iat_min') else None, 0), errors='coerce').fillna(0) if _any_col_contains(cols, 'iat_min') else 0.0
        out['iat_var'] = pd.to_numeric(df.get(_any_col_contains(cols, 'iat_var') if _any_col_contains(cols, 'iat_var') else None, 0), errors='coerce').fillna(0) if _any_col_contains(cols, 'iat_var') else 0.0
        out['iat_skew'] = pd.to_numeric(df.get(_any_col_contains(cols, 'iat_skew') if _any_col_contains(cols, 'iat_skew') else None, 0), errors='coerce').fillna(0) if _any_col_contains(cols, 'iat_skew') else 0.0
        out['iat_kurtosis'] = pd.to_numeric(df.get(_any_col_contains(cols, 'iat_kurtosis') if _any_col_contains(cols, 'iat_kurtosis') else None, 0), errors='coerce').fillna(0) if _any_col_contains(cols, 'iat_kurtosis') else 0.0

        # pkt len stats
        out['pkt_len_mean'] = pd.to_numeric(df.get(pkt_mean_col, 0), errors='coerce').fillna(0) if pkt_mean_col else out['avg_pkt_size']
        out['pkt_len_max'] = pd.to_numeric(df.get(pkt_max_col, 0), errors='coerce').fillna(0) if pkt_max_col else out['avg_pkt_size']
        out['pkt_len_min'] = pd.to_numeric(df.get(pkt_min_col, 0), errors='coerce').fillna(0) if pkt_min_col else out['avg_pkt_size']
        out['pkt_len_std'] = pd.to_numeric(df.get(pkt_std_col, 0), errors='coerce').fillna(0) if pkt_std_col else 0.0
        out['pkt_len_var'] = out['pkt_len_std'] ** 2
        out['pkt_len_skew'] = pd.to_numeric(df.get(_any_col_contains(cols, 'pkt_len_skew') if _any_col_contains(cols, 'pkt_len_skew') else None, 0), errors='coerce').fillna(0) if _any_col_contains(cols, 'pkt_len_skew') else 0.0
        out['pkt_len_kurtosis'] = pd.to_numeric(df.get(_any_col_contains(cols, 'pkt_len_kurtosis') if _any_col_contains(cols, 'pkt_len_kurtosis') else None, 0), errors='coerce').fillna(0) if _any_col_contains(cols, 'pkt_len_kurtosis') else 0.0

        # direction ratios
        out['fwd_bwd_ratio_packets'] = _safe_div(out['fwd_packets'], out['bwd_packets'])
        out['fwd_bwd_ratio_bytes'] = _safe_div(out['fwd_bytes'], out['bwd_bytes'])
        out['bytes_per_pkt'] = _safe_div(out['bytes'], out['packets'])
        out['pkts_per_byte'] = _safe_div(out['packets'], out['bytes'])

        # entropy
        out['ip_dst_entropy'] = pd.to_numeric(df.get(ip_dst_entropy_col, 0), errors='coerce').fillna(0) if ip_dst_entropy_col else 0.0
        out['port_dst_entropy'] = pd.to_numeric(df.get(port_dst_entropy_col, 0), errors='coerce').fillna(0) if port_dst_entropy_col else out['port_entropy']

        # flag ratios
        out['ack_ratio'] = pd.to_numeric(df.get(ack_col, 0), errors='coerce').fillna(0) / packets_num if ack_col else 0.0
        out['psh_ratio'] = pd.to_numeric(df.get(psh_col, 0), errors='coerce').fillna(0) / packets_num if psh_col else 0.0
        out['urg_ratio'] = pd.to_numeric(df.get(urg_col, 0), errors='coerce').fillna(0) / packets_num if urg_col else 0.0
        out['ece_ratio'] = pd.to_numeric(df.get(ece_col, 0), errors='coerce').fillna(0) / packets_num if ece_col else 0.0
        out['cwr_ratio'] = pd.to_numeric(df.get(cwr_col, 0), errors='coerce').fillna(0) / packets_num if cwr_col else 0.0

        # burst / activity
        out['flow_active_time'] = pd.to_numeric(df.get(active_col, 0), errors='coerce').fillna(0) if active_col else 0.0
        out['flow_idle_time'] = pd.to_numeric(df.get(idle_col, 0), errors='coerce').fillna(0) if idle_col else 0.0

        # throughput proxies
        out['bytes_per_second'] = out['bps']
        out['packets_per_second'] = out['pps']

        # обеспечение наличия всех колонок
        for col in ALL_FEATURES:
            if col not in out.columns:
                out[col] = 0.0

        # labels
        label_col = next((c for c in df.columns if c.lower() == 'label' or 'label' in c.lower()), None)
        if label_col:
            labels = df[label_col].fillna('Benign')
        else:
            labels = pd.Series(['Benign'] * len(df), index=df.index)
        labels = labels.astype(str).str.lower().apply(lambda x: 0 if 'benign' in x or x == '0' else 1)

        result_df = out[ALL_FEATURES].copy()
        result_df['label'] = labels.values
        # очистка
        result_df = result_df.replace([np.inf, -np.inf], 0).fillna(0)
        return result_df.reset_index(drop=True)

    # ---------------- packet-level branch ----------------
    # стандартизация названий
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    df = create_flow_key(df)

    if 'timestamp' not in df.columns or df['timestamp'].isna().all():
        df['timestamp'] = np.arange(len(df))

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0)
    df = df.sort_values('timestamp').reset_index(drop=True)

    results = []
    # определяем возможные имена столбцов в пакете (кандидаты)
    pkt_len_col_candidate = _any_col_contains(df.columns, 'pkt_len') or _any_col_contains(df.columns, 'packet_length') or _any_col_contains(df.columns, 'length') or _any_col_contains(df.columns, 'pkt_len') or None
    dst_port_col = _any_col_contains(df.columns, 'dst_port') or _any_col_contains(df.columns, 'destination_port') or None
    dst_ip_col = _any_col_contains(df.columns, 'dst_ip') or _any_col_contains(df.columns, 'destination_ip') or None
    syn_flag_col = _any_col_contains(df.columns, 'syn_flag') or _any_col_contains(df.columns, 'syn')
    rst_flag_col = _any_col_contains(df.columns, 'rst_flag') or _any_col_contains(df.columns, 'rst')
    fin_flag_col = _any_col_contains(df.columns, 'fin_flag') or _any_col_contains(df.columns, 'fin')
    ack_flag_col = _any_col_contains(df.columns, 'ack_flag') or _any_col_contains(df.columns, 'ack')
    psh_flag_col = _any_col_contains(df.columns, 'psh_flag') or _any_col_contains(df.columns, 'psh')
    urg_flag_col = _any_col_contains(df.columns, 'urg_flag') or _any_col_contains(df.columns, 'urg')
    ece_flag_col = _any_col_contains(df.columns, 'ece_flag') or _any_col_contains(df.columns, 'ece')
    cwr_flag_col = _any_col_contains(df.columns, 'cwr_flag') or _any_col_contains(df.columns, 'cwr')

    for flow_key, group in df.groupby('flow_key'):
        if len(group) == 0:
            continue

        duration = group['timestamp'].max() - group['timestamp'].min()
        if duration <= 0:
            duration = 0.001

        packets = len(group)

        # bytes (попробуем взять TotLen поля если есть)
        fwd_bytes_val = group['TotLen_Fwd_Pkts'].sum() if 'TotLen_Fwd_Pkts' in group.columns else 0.0
        bwd_bytes_val = group['TotLen_Bwd_Pkts'].sum() if 'TotLen_Bwd_Pkts' in group.columns else 0.0
        bytes_total = fwd_bytes_val + bwd_bytes_val
        if bytes_total == 0:
            # попытаемся использовать pkt_len колонку
            if pkt_len_col_candidate in group.columns:
                bytes_total = group[pkt_len_col_candidate].sum()
            else:
                bytes_total = packets * 1000  # безопасный proxy

        fwd_packets = packets // 2
        bwd_packets = packets - fwd_packets
        fwd_bytes = bytes_total // 2
        bwd_bytes = bytes_total - fwd_bytes

        # flags
        syn = group[syn_flag_col].sum() if syn_flag_col and syn_flag_col in group.columns else (group['SYN_Flag_Count'].sum() if 'SYN_Flag_Count' in group.columns else 0)
        rst = group[rst_flag_col].sum() if rst_flag_col and rst_flag_col in group.columns else (group['RST_Flag_Count'].sum() if 'RST_Flag_Count' in group.columns else 0)
        fin = group[fin_flag_col].sum() if fin_flag_col and fin_flag_col in group.columns else (group['FIN_Flag_Count'].sum() if 'FIN_Flag_Count' in group.columns else 0)

        ack = group[ack_flag_col].sum() if ack_flag_col and ack_flag_col in group.columns else 0
        psh = group[psh_flag_col].sum() if psh_flag_col and psh_flag_col in group.columns else 0
        urg = group[urg_flag_col].sum() if urg_flag_col and urg_flag_col in group.columns else 0
        ece = group[ece_flag_col].sum() if ece_flag_col and ece_flag_col in group.columns else 0
        cwr = group[cwr_flag_col].sum() if cwr_flag_col and cwr_flag_col in group.columns else 0

        # IAT series
        iat = group['timestamp'].diff().dropna()
        iat_mean, iat_max, iat_min, iat_var, iat_skew, iat_kurtosis = _series_stats(iat)

        # pkt_len stats
        if pkt_len_col_candidate and pkt_len_col_candidate in group.columns:
            pkt_s = pd.to_numeric(group[pkt_len_col_candidate], errors='coerce').dropna()
            pkt_mean, pkt_max, pkt_min, pkt_var, pkt_skew, pkt_kurt = _series_stats(pkt_s)
            pkt_std = np.sqrt(pkt_var)
        else:
            pkt_mean, pkt_max, pkt_min, pkt_var, pkt_skew, pkt_kurt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            pkt_std = 0.0

        # entropy of dst_port / dst_ip within this flow (useful if flow contains multiple dst ports/ips)
        if dst_port_col and dst_port_col in group.columns and group[dst_port_col].nunique() > 1:
            port_entropy_val = entropy(group[dst_port_col].value_counts(normalize=True) + 1e-10)
        else:
            port_entropy_val = 0.0

        if dst_ip_col and dst_ip_col in group.columns and group[dst_ip_col].nunique() > 1:
            ip_entropy_val = entropy(group[dst_ip_col].value_counts(normalize=True) + 1e-10)
        else:
            ip_entropy_val = 0.0

        # activity proxies (simple heuristics)
        # active_time: total time where inter-arrival < mean_iat * 2
        active_mask = (group['timestamp'].diff().fillna(0) < max(0.001, iat_mean * 2)) if len(group) > 1 else pd.Series([False] * len(group))
        flow_active_time = active_mask.sum()  # count of packets in active periods (proxy)
        flow_idle_time = duration - flow_active_time

        row = {
            'duration': float(duration),
            'packets': int(packets),
            'bytes': float(bytes_total),
            'pps': packets / duration,
            'bps': bytes_total / duration,
            'avg_pkt_size': (bytes_total / packets) if packets > 0 else 0.0,
            'port_entropy': float(port_entropy_val),
            'iat_std': float(iat_var) ** 0.5 if iat_var >= 0 else 0.0,
            'unique_dst_port': int(group[dst_port_col].nunique()) if dst_port_col and dst_port_col in group.columns else 1,
            'unique_dst_ip': int(group[dst_ip_col].nunique()) if dst_ip_col and dst_ip_col in group.columns else 1,
            'syn_ratio': syn / packets if packets > 0 else 0.0,
            'rst_ratio': rst / packets if packets > 0 else 0.0,
            'fin_ratio': fin / packets if packets > 0 else 0.0,
            'fwd_packets': int(fwd_packets),
            'bwd_packets': int(bwd_packets),
            'fwd_bytes': float(fwd_bytes),
            'bwd_bytes': float(bwd_bytes),

            # новые
            'iat_mean': float(iat_mean),
            'iat_max': float(iat_max),
            'iat_min': float(iat_min),
            'iat_var': float(iat_var),
            'iat_skew': float(iat_skew),
            'iat_kurtosis': float(iat_kurtosis),

            'pkt_len_mean': float(pkt_mean),
            'pkt_len_max': float(pkt_max),
            'pkt_len_min': float(pkt_min),
            'pkt_len_std': float(pkt_std),
            'pkt_len_var': float(pkt_var),
            'pkt_len_skew': float(pkt_skew),
            'pkt_len_kurtosis': float(pkt_kurt),

            'fwd_bwd_ratio_packets': _safe_div(fwd_packets, bwd_packets),
            'fwd_bwd_ratio_bytes': _safe_div(fwd_bytes, bwd_bytes),
            'bytes_per_pkt': _safe_div(bytes_total, packets),
            'pkts_per_byte': _safe_div(packets, bytes_total),

            'ip_dst_entropy': float(ip_entropy_val),
            'port_dst_entropy': float(port_entropy_val),

            'ack_ratio': ack / packets if packets > 0 else 0.0,
            'psh_ratio': psh / packets if packets > 0 else 0.0,
            'urg_ratio': urg / packets if packets > 0 else 0.0,
            'ece_ratio': ece / packets if packets > 0 else 0.0,
            'cwr_ratio': cwr / packets if packets > 0 else 0.0,

            'flow_active_time': float(flow_active_time),
            'flow_idle_time': float(flow_idle_time),

            'bytes_per_second': bytes_total / duration,
            'packets_per_second': packets / duration
        }

        results.append(row)

    result_df = pd.DataFrame(results)

    # --- labels ---
    if 'label' in df.columns:
        label_map = df[['flow_key', 'label']].drop_duplicates('flow_key').set_index('flow_key')['label']
        result_df = result_df.merge(label_map, left_index=True, right_index=True, how='left')
        result_df['label'] = result_df['label'].fillna('Benign')
    else:
        result_df['label'] = 'Benign'

    result_df['label'] = result_df['label'].astype(str).str.lower()
    result_df['label'] = result_df['label'].apply(lambda x: 0 if 'benign' in x or x == '0' else 1)

    # === 6. Заполняем недостающие колонки ===
    for col in ALL_FEATURES:
        if col not in result_df.columns:
            result_df[col] = 0.0

    # === 7. Очистка от NaN и бесконечностей ===
    result_df = result_df.replace([np.inf, -np.inf], 0).fillna(0)

    return result_df[ALL_FEATURES + ['label']]


# Для совместимости
extract_features = extract_flow_level_features
