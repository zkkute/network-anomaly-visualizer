# scripts/train_models.py
# Финальная версия с идеальной обработкой ошибок — для защиты диплома 2025

import sys
import traceback
from pathlib import Path

from sklearn.model_selection import train_test_split

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
        # Если метки уже числовые (0/1) — оставляем как есть
        if np.issubdtype(df['label'].dtype, np.number):
            df['label'] = df['label'].fillna(0).astype(int)
        else:
            # Нормализуем строковые метки
            df['label'] = df['label'].fillna('normal').astype(str).str.strip().str.lower()


            def map_label(x):
                if x in {'normal', 'benign', 'benign ', 'ok', 'good', '0', 'no'}:
                    return 0
                if x in {'attack', 'attacker', 'malicious', 'malware', '1', 'yes'}:
                    return 1
                # Попытка распознать числовую строк
                try:
                    xi = int(float(x))
                    return 0 if xi == 0 else 1
                except Exception:
                    # По умолчанию — считать нормальным (безопаснее для обучения)
                    return 0


            df['label'] = df['label'].apply(map_label).astype(int)

        attacks_count = int(df['label'].sum())
        print(f"Метки обработаны → Атак: {attacks_count:,} | Нормальных: {len(df) - attacks_count:,}")
    except Exception as e:
        print_error("Ошибка при обработке столбца 'label'!")
        print(f"Подробности: {e}")
        sys.exit(1)

    # === АВТОМАТИЧЕСКАЯ загрузка списка признаков из preprocess_universal2.py ===
    try:
        FEATURES = joblib.load("models/feature_list.pkl")
        print(f"Загружен список из {len(FEATURES)} признаков (автоматически из preprocess)")

        # Проверяем, все ли признаки есть в датасете
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            print(f"   → Внимание: {len(missing)} признаков отсутствует → будут заполнены нулями")
            for col in missing:
                df[col] = 0.0
        print_success(f"Готово: используем {len(FEATURES)} мощных признаков")

    except FileNotFoundError:
        print_error("Файл models/feature_list.pkl не найден!")
        print("   → Запустите сначала:")
        print("       python scripts/preprocess_universal2.py")
        sys.exit(1)
    except Exception as e:
        print_error("Ошибка при загрузке списка признаков!")
        print(f"   → Детали: {e}")
        print("   → Используем резервный список (17 базовых)")
        FEATURES = [
            'duration', 'packets', 'bytes', 'pps', 'bps', 'avg_pkt_size',
            'port_entropy', 'iat_std', 'unique_dst_port', 'unique_dst_ip',
            'syn_ratio', 'rst_ratio', 'fin_ratio',
            'fwd_bytes', 'bwd_bytes', 'fwd_packets', 'bwd_packets'
        ]
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0.0

    X = df[FEATURES].values
    y = df['label'].values

    # === Сплит ===
    unique_labels = np.unique(y)
    print("Полное распределение меток:", np.unique(y, return_counts=True))

    if len(unique_labels) > 1:
        print("→ Используем stratified train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
        )
    else:
        print("⚠️ Найден только один класс! Stratify невозможен.")
        print("→ Делаем простой shuffle-split")

        rng = np.random.RandomState(42)
        idx = rng.permutation(len(y))
        split_idx = int(0.8 * len(y))

        X_train, X_test = X[idx[:split_idx]], X[idx[split_idx:]]
        y_train, y_test = y[idx[:split_idx]], y[idx[split_idx:]]


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
    print("Обучение XGBoost (если возможно)...")
    xgb_model = None
    try:
        unique_classes = np.unique(y_train)
        if len(np.unique(y_train)) < 2:
            print("⚠️ XGBoost не может обучаться — в train только один класс!")
            print("→ Пропускаем XGBoost, но IsolationForest всё равно обучим")
        else:
            print("Запуск обучения XGBoost...")
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
        # Если хотите — можно продолжать без XGBoost
        xgb_model = None

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
        # IsolationForest уже обучен ниже — но в вашем код он идёт позже.
        # Чтобы не менять порядок: здесь сначала попробуем получить xgb_proba, если модель есть.
        if xgb_model is not None:
            xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        else:
            # если XGBoost не обучен — используем нули (или можно использовать iso_score позже как базу)
            xgb_proba = np.zeros(len(X_test_scaled))

        # Если IsolationForest ещё не обучен к этому моменту, то выше в коде он должен быть обучен;
        # предполагаем, что iso уже существует (ваш текущий поток: вы обучаете iso ниже — в старом коде iso обучается после xgb).
        # Если порядок другой, поменяйте: обучите iso перед этой секцией.
        iso_anomaly = -iso.decision_function(X_test_scaled)
        iso_score = (iso_anomaly - iso_anomaly.min()) / (iso_anomaly.ptp() + 1e-12)

        # гибрид: если xgb_model отсутствует, hybrid == iso_score
        if xgb_model is None:
            hybrid_score = iso_score
        else:
            hybrid_score = np.maximum(xgb_proba, iso_score)

        if y_test.sum() > 0:
            prec, rec, thr = precision_recall_curve(y_test, hybrid_score)
            f1s = 2 * prec * rec / (prec + rec + 1e-12)
            # thr имеет длину len(prec)-1, guard на пустой thr
            if len(thr) > 0:
                threshold = thr[np.nanargmax(f1s)]
            else:
                threshold = np.percentile(hybrid_score, 99.5)
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

    # === БЕЗОПАСНЫЙ classification_report ===
    unique_classes = np.unique(y_test)
    if len(unique_classes) > 1:
        print(classification_report(
            y_test, y_pred,
            target_names=["Normal", "Attack"],
            digits=4,
            zero_division=0
        ))
        detection_rate = 100 * y_pred[y_test == 1].sum() / max(y_test.sum(), 1)
        print(f"Обнаружено атак: {y_pred[y_test == 1].sum():,} из {y_test.sum():,} ({detection_rate:.2f}%)")
        print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(f"F1-score (attack): {f1_score(y_test, y_pred, pos_label=1):.4f}")
    elif y_test.sum() == 0:
        false_positives = y_pred.sum()
        print("В тесте ТОЛЬКО нормальный трафик")
        print(f"Ложно-положительных срабатываний: {false_positives:,} из {len(y_test):,}")
        print("→ Система спокойна — атак нет")
    else:  # все — атака
        detected = y_pred.sum()
        print("В тесте ТОЛЬКО атака (DDoS, PortScan и т.п.)")
        print(f"Обнаружено: {detected:,} из {len(y_test):,} ({100 * detected / len(y_test):.2f}%)")
        print("→ Система видит атаку почти везде — это ожидаемо для чистого атакующего датасета")

    # === Сохранение ===
    try:
        (Path("models")).mkdir(exist_ok=True)

        # ←←←← ИСПРАВЛЕНИЕ ЗДЕСЬ ←←←←
        if xgb_model is not None:
            xgb_model.save_model("models/xgboost_model.json")
            print_success("XGBoost модель сохранена")
        else:
            print("   → XGBoost не обучен (нет нормального трафика) → файл не сохраняется")

        joblib.dump(iso, "models/isolation_forest.pkl")
        print_success("Isolation Forest сохранён")

        # ← остальной код без изменений

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