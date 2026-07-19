"""
Генератор реалистичных демо-данных для иллюстраций в ВКР.

Создаёт два CSV-файла:
  1. data/demo/flows_demo_for_sankey.csv     — для sankey_anomaly.py
  2. data/demo/flows_demo_for_histogram.csv  — для duration_histogram.py

Сценарий: реалистичная корпоративная сеть с активной атакой
  - Норма: веб-трафик, DNS, внутренние коммуникации
  - Атака 1: ЭКСФИЛЬТРАЦИЯ — большой объём с 192.168.10.0/24 на внешний IP
    (даст яркую красную связь на Sankey - именно то, что описано в разделе 4.7 ВКР)
  - Атака 2: C&C-канал — заражённая машина мелкими долгими сессиями
    (даст хвост долгоживущих сессий на гистограмме - раздел 4.7, сценарий 5)
  - Атака 3: Brute force на SSH-сервер
  - Атака 4: DoS-всплеск против веб-сервера

Запуск:
  python scripts/generate_demo_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SANKEY_PATH = OUT_DIR / "flows_demo_for_sankey.csv"
HIST_PATH = OUT_DIR / "flows_demo_for_histogram.csv"

RNG = np.random.default_rng(seed=2024)


# ===================== ПОМОЩНИКИ =====================

def random_ip(subnet_prefix, n=1):
    """Генерирует случайные IP в подсети. subnet_prefix = '192.168.10' для /24"""
    return [f"{subnet_prefix}.{RNG.integers(1, 254)}" for _ in range(n)]


def proto_from_port(port):
    if port in (80, 8080): return "HTTP"
    if port in (443, 8443): return "HTTPS"
    if port == 53: return "DNS"
    if port == 22: return "SSH"
    if port == 21: return "FTP"
    return "TCP/UDP"


# ===================== ГЕНЕРАЦИЯ ПОТОКОВ =====================

def generate_normal_web(n):
    """Обычный HTTP/HTTPS трафик: рабочие станции -> внешние сайты"""
    rows = []
    for _ in range(n):
        src = f"192.168.{RNG.choice([10, 11, 12])}.{RNG.integers(1, 254)}"
        # Реалистичные внешние подсети
        dst_subnet = RNG.choice(["142.250.190", "151.101.1", "104.16.132", "13.107.42"])
        dst = f"{dst_subnet}.{RNG.integers(1, 254)}"
        port = RNG.choice([80, 443, 443, 443])  # HTTPS преобладает
        dur = float(RNG.lognormal(0.5, 1.0))  # медиана ~1.6 сек
        bytes_v = int(RNG.lognormal(8.5, 1.5))  # средние объёмы
        score = float(np.clip(RNG.beta(1.2, 12), 0, 0.4))  # норма
        rows.append({
            "src_ip": src, "dst_ip": dst,
            "src_port": int(RNG.integers(49152, 65535)),
            "dst_port": int(port),
            "protocol": proto_from_port(port),
            "duration": dur, "bytes": bytes_v, "score": score,
            "label": 0, "attack_type": "normal",
        })
    return rows


def generate_dns(n):
    """DNS-запросы: рабочие станции -> внутренний DNS-сервер"""
    rows = []
    for _ in range(n):
        src = f"192.168.{RNG.choice([10, 11, 12])}.{RNG.integers(1, 254)}"
        dst = f"192.168.1.{RNG.choice([1, 53])}"  # DNS-сервер
        dur = float(RNG.lognormal(-2.5, 0.6))  # очень короткие, ~0.08 сек
        bytes_v = int(RNG.uniform(80, 350))  # маленькие
        score = float(np.clip(RNG.beta(1, 15), 0, 0.3))
        rows.append({
            "src_ip": src, "dst_ip": dst,
            "src_port": int(RNG.integers(49152, 65535)),
            "dst_port": 53,
            "protocol": "DNS",
            "duration": dur, "bytes": bytes_v, "score": score,
            "label": 0, "attack_type": "normal",
        })
    return rows


def generate_internal(n):
    """Внутренние коммуникации: рабочие станции -> сервера в 192.168.20.0/24"""
    rows = []
    for _ in range(n):
        src = f"192.168.{RNG.choice([10, 11, 12])}.{RNG.integers(1, 254)}"
        dst = f"192.168.20.{RNG.integers(10, 50)}"
        port = RNG.choice([445, 3389, 80, 443, 8080])  # SMB, RDP, веб
        dur = float(RNG.lognormal(1.0, 1.2))  # медиана ~2.7 сек
        bytes_v = int(RNG.lognormal(9.5, 1.8))
        score = float(np.clip(RNG.beta(1.2, 10), 0, 0.45))
        rows.append({
            "src_ip": src, "dst_ip": dst,
            "src_port": int(RNG.integers(49152, 65535)),
            "dst_port": int(port),
            "protocol": proto_from_port(port) if port in (80, 443, 8080) else "TCP/UDP",
            "duration": dur, "bytes": bytes_v, "score": score,
            "label": 0, "attack_type": "normal",
        })
    return rows


def generate_exfiltration(n):
    """АТАКА 1: ЭКСФИЛЬТРАЦИЯ
    Заражённая машина из 192.168.10.0/24 -> внешний C&C-адрес.
    Признаки: БОЛЬШОЙ объём, мало потоков, долго, высокий score.
    Именно эта атака описана в сценарии 3 раздела 4.7 ВКР.
    """
    rows = []
    # ОДИН заражённый источник и ОДИН внешний IP - так выглядит реальная эксфильтрация
    infected_host = f"192.168.10.{RNG.integers(50, 100)}"
    c2_server = f"203.0.113.{RNG.integers(10, 30)}"
    for _ in range(n):
        dur = float(RNG.lognormal(4.0, 0.6))  # долго, ~55 сек медиана
        bytes_v = int(RNG.lognormal(13.5, 0.5))  # ОЧЕНЬ много байт (~700KB-2MB)
        score = float(np.clip(RNG.uniform(0.82, 0.97), 0, 1))
        rows.append({
            "src_ip": infected_host, "dst_ip": c2_server,
            "src_port": int(RNG.integers(49152, 65535)),
            "dst_port": 443,  # маскировка под HTTPS
            "protocol": "HTTPS",
            "duration": dur, "bytes": bytes_v, "score": score,
            "label": 1, "attack_type": "exfiltration",
        })
    return rows


def generate_c2_beacons(n):
    """АТАКА 2: C&C beacons
    Регулярные мелкие соединения с управляющим сервером.
    Признаки: МНОГО коротких сессий, МАЛЕНЬКИЙ объём, ПОСТОЯННЫЙ длительный паттерн.
    Именно этот паттерн упомянут в сценарии 5 раздела 4.7 ВКР как
    «долгоживущие соединения, потенциально связанные с C&C».
    """
    rows = []
    botnet_hosts = [f"192.168.11.{i}" for i in RNG.integers(20, 250, size=3)]
    c2_servers = [f"185.220.{RNG.integers(100, 110)}.{RNG.integers(1, 254)}" for _ in range(2)]
    for _ in range(n):
        src = RNG.choice(botnet_hosts)
        dst = RNG.choice(c2_servers)
        # ДЛИННЫЕ соединения с минимальной активностью - это даст хвост на гистограмме
        dur = float(RNG.lognormal(5.5, 0.8))  # медиана ~240 сек, до часа
        bytes_v = int(RNG.lognormal(7.0, 0.8))  # очень мало байт - keep-alive
        score = float(np.clip(RNG.uniform(0.72, 0.88), 0, 1))
        rows.append({
            "src_ip": src, "dst_ip": dst,
            "src_port": int(RNG.integers(49152, 65535)),
            "dst_port": RNG.choice([443, 8443]),
            "protocol": "HTTPS",
            "duration": dur, "bytes": bytes_v, "score": score,
            "label": 1, "attack_type": "c2_beacon",
        })
    return rows


def generate_brute_force(n):
    """АТАКА 3: SSH brute force
    Атакующий снаружи -> SSH-сервер с множеством коротких попыток
    """
    rows = []
    attacker = f"45.142.{RNG.integers(120, 130)}.{RNG.integers(1, 254)}"
    ssh_server = f"192.168.20.{RNG.choice([22, 25])}"
    for _ in range(n):
        dur = float(RNG.lognormal(-0.5, 0.6))  # ~0.6 сек, короткие попытки
        bytes_v = int(RNG.uniform(150, 800))
        score = float(np.clip(RNG.uniform(0.65, 0.92), 0, 1))
        rows.append({
            "src_ip": attacker, "dst_ip": ssh_server,
            "src_port": int(RNG.integers(49152, 65535)),
            "dst_port": 22,
            "protocol": "SSH",
            "duration": dur, "bytes": bytes_v, "score": score,
            "label": 1, "attack_type": "brute_force",
        })
    return rows


def generate_dos_flood(n):
    """АТАКА 4: DoS-всплеск на веб-сервер
    Множество источников -> один сервер за КОРОТКОЕ время
    """
    rows = []
    web_server = f"192.168.20.{RNG.choice([80, 81])}"
    for _ in range(n):
        # Источники - "случайные" внешние адреса (имитация botnet)
        src = f"{RNG.integers(1, 254)}.{RNG.integers(1, 254)}.{RNG.integers(1, 254)}.{RNG.integers(1, 254)}"
        dur = float(RNG.lognormal(-3.0, 0.5))  # МИКРОсессии ~0.05 сек
        bytes_v = int(RNG.uniform(60, 200))  # очень маленькие
        score = float(np.clip(RNG.uniform(0.78, 0.95), 0, 1))
        rows.append({
            "src_ip": src, "dst_ip": web_server,
            "src_port": int(RNG.integers(1024, 65535)),
            "dst_port": 80,
            "protocol": "HTTP",
            "duration": dur, "bytes": bytes_v, "score": score,
            "label": 1, "attack_type": "dos",
        })
    return rows


# ===================== СБОРКА ДВУХ ДАТАСЕТОВ =====================

def build_sankey_dataset():
    """Датасет для Sankey: акцент на разнообразии направлений и аномалиях"""
    print("[Sankey] Генерирую потоки...")
    rows = []
    rows += generate_normal_web(1800)  # 36% — фон
    rows += generate_dns(800)  # 16% — DNS
    rows += generate_internal(1200)  # 24% — внутренние
    rows += generate_exfiltration(80)  # 1.6% — но БОЛЬШОЙ объём ← красная связь
    rows += generate_c2_beacons(60)  # 1.2% — длинные мелкие
    rows += generate_brute_force(150)  # 3% — SSH brute
    rows += generate_dos_flood(900)  # 18% — DoS

    df = pd.DataFrame(rows)
    # Перемешать, чтобы не было идеального порядка
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(SANKEY_PATH, index=False)
    print(f"[Sankey] Сохранено: {SANKEY_PATH} ({len(df):,} потоков)")
    print(f"         Из них аномалий: {int(df['label'].sum()):,}")
    print(f"         Объём эксфильтрации: {df[df['attack_type'] == 'exfiltration']['bytes'].sum():,} байт")
    return df


def build_histogram_dataset():
    """Датасет для гистограммы: акцент на бимодальном распределении длительности"""
    print("\n[Histogram] Генерирую потоки...")
    rows = []
    # Основная масса — короткий веб и DNS
    rows += generate_normal_web(15000)
    rows += generate_dns(8000)
    rows += generate_internal(5500)
    # Хвост — долгоживущие C&C-сессии (то самое, что нужно показать в разделе 4.7)
    rows += generate_c2_beacons(400)
    # Немного DoS — для острого пика слева
    rows += generate_dos_flood(800)
    # Эксфильтрация — длинные жирные сессии
    rows += generate_exfiltration(150)
    # SSH brute — короткие
    rows += generate_brute_force(200)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(HIST_PATH, index=False)
    print(f"[Histogram] Сохранено: {HIST_PATH} ({len(df):,} потоков)")
    print(f"            Из них аномалий: {int(df['label'].sum()):,}")
    print(f"            Длительность: медиана={df['duration'].median():.2f}с, "
          f"p95={df['duration'].quantile(0.95):.1f}с, "
          f"max={df['duration'].max():.1f}с")
    return df


if __name__ == "__main__":
    build_sankey_dataset()
    build_histogram_dataset()
    print("\nГотово! Теперь запускайте:")
    print("  python scripts/sankey_anomaly.py --input data/demo/flows_demo_for_sankey.csv")
    print("  python scripts/duration_histogram.py --input data/demo/flows_demo_for_histogram.csv")