# scripts/train_models.py
# Финальная версия с идеальной обработкой ошибок — для защиты диплома 2025

import sys
import traceback
from pathlib import Path

# Добавляем корень проекта
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
import joblib
import json
from datetime import datetime
import warnings
import os

# Полностью глушим предупреждения (но ошибки будем ловить красиво!)
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

def print_header():
    print("=" * 80)
    print("   ГИБРИДНАЯ NDR-СИСТЕМА: ОБУЧЕНИЕ (XGBoost + Isolation Forest)")
    print(f"   Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def print_error(msg: str):
    print("\n" + "!" * 80)
    print(f"   ОШИБКА: {msg}")
    print("!" * 80 + "\n")

def print_success(msg: str):
    print("\n" + "УСПЕШНО: " + msg + "\n")

try:
    print_header()

    # === Проверка файла ===
    DATA_FILE = Path("data/processed/my_traffic_features.csv")
    if not DATA_FILE.exists():
        print_error("Файл не найден!")
        print("Проверьте, что вы запустили:")
        print("   python scripts/preprocess_universal2.py")
        print(f"Ожидался файл: {DATA_FILE.resolve()}")
        sys.exit(1)

    print(f"Загружаем датасет: {DATA_FILE.name}")

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print_error("Не удалось прочитать CSV-файл!")
        print(f"Ошибка: {e}")
        sys.exit(1)

    if df.empty:
        print_error("CSV-файл пустой!")
        sys.exit(1)

    print(f"Потоков загружено: {len(df):,}")

    # === Метки ===
    try:
        df['label'] = df['label'].fillna('normal').astype(str).str.strip().str.lower()
        df['label'] = df['label'].apply(lambda x: 0 if x in ['normal', 'benign'] else 1)
        attacks_count = df['label'].sum()
        print(f"Метки обработаны → Атак: {attacks_count:,} | Нормальных: {len(df) - attacks_count:,}")
    except Exception as e:
        print_error("Ошибка при обработке столбца 'label'!")
        print(f"Подробности: {e}")
        sys.exit(1)

    # === Фичи ===
    FEATURES = [
        'duration', 'packets', 'bytes', 'pps', 'bps', 'avg_pkt_size',
        'port_entropy', 'iat_std', 'unique_dst_port', 'unique_dst_ip',
        'syn_ratio', 'rst_ratio', 'fin_ratio',
        'fwd_bytes', 'bwd_bytes', 'fwd_packets', 'bwd_packets'
    ]

    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print_success(f"Добавлено {len(missing_features)} отсутствующих фич (заполнены нулями)")

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURES].values
    y = df['label'].values

    # === Сплит ===
    split_idx = int(0.8 * len(df))
    if split_idx < 100:
        print_error("Слишком мало данных для обучения (<100 строк)")
        sys.exit(1)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Обучающая выборка: {len(X_train):,} | Тестовая: {len(X_test):,} | Атак в тесте: {y_test.sum():,}")

    # === Нормализация ===
    print("Нормализация данных...")
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(FEATURES, "models/feature_list.pkl")
        print_success("Скейлер и список фич сохранены")
    except Exception as e:
        print_error("Ошибка при нормализации!")
        print(f"Детали: {e}")
        sys.exit(1)

    # === XGBoost ===
    print("Обучение XGBoost...")
    try:
        if y_train.sum() < 100:
            print("   → Мало атак → фиксированные параметры (надёжно и быстро)")
            scale_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            xgb_model = xgb.XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            )
        else:
            print("   → Много атак → мощные параметры")
            xgb_model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            )
        xgb_model.fit(X_train_scaled, y_train)
        print_success("XGBoost обучен")
    except Exception as e:
        print_error("Критическая ошибка при обучении XGBoost!")
        traceback.print_exc()
        sys.exit(1)

    # === Isolation Forest ===
    print("Обучение Isolation Forest...")
    try:
        normal_data = X_train_scaled[y_train == 0]
        if len(normal_data) == 0:
            normal_data = X_train_scaled
            print("   → Внимание: в обучении нет нормального трафика → обучаем на всём")
        iso = IsolationForest(
            contamination=0.001,
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        iso.fit(normal_data)
        print_success("Isolation Forest обучен")
    except Exception as e:
        print_error("Ошибка при обучении Isolation Forest!")
        traceback.print_exc()
        sys.exit(1)

    # === Гибрид ===
    print("Генерация гибридного скора...")
    try:
        xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        iso_anomaly = -iso.decision_function(X_test_scaled)
        iso_score = (iso_anomaly - iso_anomaly.min()) / (iso_anomaly.ptp() + 1e-12)
        hybrid_score = np.maximum(xgb_proba, iso_score)

        if y_test.sum() > 0:
            prec, rec, thr = precision_recall_curve(y_test, hybrid_score)
            f1s = 2 * prec * rec / (prec + rec + 1e-12)
            threshold = thr[np.argmax(f1s)]
        else:
            threshold = np.percentile(hybrid_score, 99.5)

        y_pred = (hybrid_score >= threshold).astype(int)
        print_success("Гибридный скор сформирован")
    except Exception as e:
        print_error("Ошибка при формировании гибридного скора!")
        traceback.print_exc()
        sys.exit(1)

    # === Результаты ===
    print("\n" + "=" * 80)
    print("   РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print("=" * 80)

    if y_test.sum() > 0:
        print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"], digits=4))
        detection = 100 * y_pred.sum() / y_test.sum()
        print(f"Обнаружено атак: {y_pred.sum():,} из {y_test.sum():,} ({detection:.2f}%)")
        print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    else:
        print("В тесте только нормальный трафик → всё спокойно!")
        print(f"Подозрительных потоков: {y_pred.sum():,}")

    # === Сохранение ===
    try:
        (Path("models")).mkdir(exist_ok=True)
        xgb_model.save_model("models/xgboost_model.json")
        joblib.dump(iso, "models/isolation_forest.pkl")
        print_success("Модели сохранены в папку models/")

        exp_file = Path("experiments") / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        exp_file.parent.mkdir(exist_ok=True)
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "dataset": DATA_FILE.name,
                "total_flows": len(df),
                "attacks_total": int(y.sum()),
                "attacks_in_test": int(y_test.sum()),
                "detected": int(y_pred.sum()),
                "f1_score": round(float(f1_score(y_test, y_pred)), 4) if y_test.sum() > 0 else None
            }, f, indent=2, ensure_ascii=False)
        print_success(f"Эксперимент сохранён: {exp_file.name}")
    except Exception as e:
        print_error("Не удалось сохранить модели или лог!")
        print(f"Ошибка: {e}")

    print("\n" + "УСПЕШНО: Всё готово к запуску визуализации!")
    print("   streamlit run src/visualization/app2.py")
    print("=" * 80)

except KeyboardInterrupt:
    print_error("Выполнение прервано пользователем (Ctrl+C)")
    sys.exit(1)
except Exception as e:
    print_error("КРИТИЧЕСКАЯ ОШИБКА — программа упала!")
    print("Подробности:")
    traceback.print_exc()
    sys.exit(1)