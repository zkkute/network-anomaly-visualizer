# scr/preprocessing/flow_aggregator.py
def aggregate_flows(df):
    """Агрегация по 5-tuple"""
    key = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
    flows = df.groupby(key).agg({
        'timestamp': ['min', 'max', 'count'],
        'length': 'sum'
    }).reset_index()
    flows.columns = key + ['ts_start', 'ts_end', 'packets', 'bytes']
    flows['duration'] = flows['ts_end'] - flows['ts_start']
    flows['pps'] = flows['packets'] / flows['duration']
    flows['bps'] = flows['bytes'] / flows['duration']
    return flows