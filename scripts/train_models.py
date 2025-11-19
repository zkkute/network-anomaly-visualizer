# scripts/train_models.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

print("Загружаем данные...")
df = pd.read_csv("data/processed/friday_features.csv")

# Бинарная задача: normal vs аномалия (чтобы гибрид работал идеально)
df['is_anomaly'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
X = df.drop(['label', 'is_anomaly'], axis=1)
y = df['is_anomaly']

# Time-based split (80% train, 20% test) — имитируем реальный сценарий
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train)} строк, Test: {len(X_test)} строк")
print(f"Аномалий в тесте: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# 1. XGBoost (supervised)
print("\nОбучаем XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# 2. Isolation Forest (unsupervised) — только на нормальном трафике из train
print("Обучаем Isolation Forest...")
normal_train = X_train[y_train == 0]
iso_model = IsolationForest(
    contamination=0.01,
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
iso_model.fit(normal_train)
iso_score = -iso_model.decision_function(X_test)
iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-8)  # 0–1

# 3. Гибридный score = max(xgb, iso)
hybrid_score = np.maximum(xgb_proba, iso_score)
y_pred = (hybrid_score > 0.8).astype(int)

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ГИБРИДНОЙ МОДЕЛИ")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['normal', 'attack']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Сохраняем всё
os.makedirs("models", exist_ok=True)
xgb_model.save_model("models/xgboost_model.json")
joblib.dump(iso_model, "models/isolation_forest.pkl")
print("\nМодели сохранены в папку models/")

print("\nГотово! Теперь запускай визуализацию → python -m streamlit run src/visualization/app.py")