import time
import sys
from pathlib import Path

# === ФИКС ПУТИ ===
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.preprocessing import StandardScaler

# Импорт функций и констант из основного препроцессора
from scripts.preprocess_universal import (
    process_unsw_nb15,
    process_cic_ids2017,
    process_custom,
    load_and_detect_dataset,
    ALL_FEATURES,
    MODELS_DIR,
    OUTPUT_FILE
)


def main():
    print("УНИВЕРСАЛЬНЫЙ ПРЕПРОЦЕССОР v3.1")
    print("Поиск датасетов в data/raw/ и подпапках...\n")

    # Находим все CSV-файлы рекурсивно
    csv_files = sorted(
        Path("data/raw").rglob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not csv_files:
        print("CSV-файлы не найдены")
        print("Помести любой датасет в папку data/raw/ (можно в подпапки)")
        return

    print(f"Найдено датасетов: {len(csv_files)}\n")
    print("Доступные файлы (новые сверху):")
    print("   №   │ Дата и время       │ Размер     │ Имя файла")
    print("   ────┼────────────────────┼────────────┼────────────────────────────────")

    for i, path in enumerate(csv_files):
        if i >= 30:
            print(f"       │                    │            │ ... и ещё {len(csv_files) - 30} файлов")
            break
        mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(path.stat().st_mtime))
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {i+1:2} │ {mtime} │ {size_mb:7.1f} МБ │ {path.name}")

    print("   ────┴────────────────────┴────────────┴────────────────────────────────\n")

    # === УМНЫЙ ВЫБОР ===
    while True:
        choice = input("Введи номер датасета (или просто Enter — взять самый новый): ").strip()

        if choice == "":
            selected_file = csv_files[0]
            print(f"\nВыбран самый новый файл:")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            selected_file = csv_files[int(choice) - 1]
            print(f"\nВыбран файл №{choice}:")
            break
        else:
            print("Неверный номер! Попробуй снова или нажми Enter.")

    # Информация о выбранном файле
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(selected_file.stat().st_mtime))
    size_mb = selected_file.stat().st_size / (1024 * 1024)
    print(f"   → {selected_file.name}")
    print(f"   → {mtime} | {size_mb:.1f} МБ")
    print(f"   → {selected_file}\n")

    print("Запуск обработки...")
    print("=" * 70)

    # === Обработка ===
    raw_df, dataset_type = load_and_detect_dataset(selected_file)

    if dataset_type == "cic":
        df = process_cic_ids2017(raw_df)
        print("Тип датасета: CIC-IDS2017 / CSE-CIC-IDS2018")
    elif dataset_type == "unsw":
        df = process_unsw_nb15(raw_df)
        print("Тип датасета: UNSW-NB15")
    else:
        df = process_custom(raw_df)
        print("Тип датасета: Кастомный (Wireshark / свой CSV)")

    # Заполняем недостающие фичи нулями
    for f in ALL_FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    df = df[ALL_FEATURES + ['label']]

    # Нормализация
    scaler = StandardScaler()
    df[ALL_FEATURES] = scaler.fit_transform(df[ALL_FEATURES])

    # Сохранение
    df.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(ALL_FEATURES, MODELS_DIR / "feature_list.pkl")

    print("\nГОТОВО! Данные успешно обработаны")
    print(f"Потоков в датасете: {len(df):,}")
    print(f"Сохранено → {OUTPUT_FILE}")
    print("\nТеперь можно запустить:")
    print("   python scripts/train_models.py")
    print("   streamlit run src/visualization/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()