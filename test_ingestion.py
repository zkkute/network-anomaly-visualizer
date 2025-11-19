# test_ingestion.py
from src.ingestion.pcap_reader import pcap_to_flows
import os

def main():
    # Путь к тестовому PCAP
    pcap_path = "data/raw_pcap/test.pcap"
    output_csv = "data/processed/flows_test.csv"

    # Проверяем, есть ли файл
    if not os.path.exists(pcap_path):
        print(f"Ошибка: Файл {pcap_path} не найден!")
        print("Скачай тестовый PCAP:")
        print("  wget -O data/raw_pcap/test.pcap https://github.com/markbaggett/samplecap/raw/master/http.pcap")
        return

    print("Запускаем парсинг PCAP → flows...")
    df = pcap_to_flows(
        pcap_path=pcap_path,
        output_csv=output_csv,
        max_packets=1000  # ограничиваем для скорости
    )

    print("\nУСПЕШНО! Получено flows:")
    print(df[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'packets_total', 'duration']].head(10))
    print(f"\nВсего flows: {len(df)}")
    print(f"Сохранено в: {output_csv}")

if __name__ == "__main__":
    main()