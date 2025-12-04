# scr/decision/alert_generator.py
import pandas as pd
import numpy as np
from src.ml.inference import get_hybrid_score  # Создадим ниже

def generate_alerts(features_df, config):
    scores = get_hybrid_score(features_df)
    features_df['score'] = scores
    # Группировка по подсетям (упрощенно: hash IP to grid)
    features_df['subnet'] = features_df['src_ip'].str[:3].astype('category').cat.codes % 16  # 16x16
    agg = features_df.groupby('subnet').agg({
        'score': 'max',
        'pps': 'mean'
    }).reset_index()
    agg['color'] = np.where(agg['score'] > config['ml']['threshold_red'], 'red',
                            np.where(agg['score'] > config['ml']['threshold_yellow'], 'yellow', 'green'))
    return agg