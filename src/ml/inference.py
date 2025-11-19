import joblib
import xgboost as xgb
import numpy as np

xgb_model = xgb.XGBClassifier()
xgb_model.load_model('models/xgboost_model.json')
if_model = joblib.load('models/isolation_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

def get_hybrid_score(df):
    X = scaler.transform(df[['pps', 'bps', 'duration', 'packets', 'bytes', 'syn_ratio', 'port_entropy']])
    proba = xgb_model.predict_proba(X)[:, 1]
    score_if = -if_model.decision_function(X)
    score_if = score_if / score_if.max() if score_if.max() > 0 else np.zeros(len(X))
    return np.maximum(proba, score_if)