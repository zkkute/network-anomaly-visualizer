#scr/ingestion/live_capture.py
import pyshark
import queue
import threading
import time
from typing import Dict, Any
import asyncio

packet_queue = queue.Queue(maxsize=30000)
_capture_thread = None
_lock = threading.Lock()


def packet_handler(pkt) -> Dict[str, Any]:
    try:
        if not hasattr(pkt, 'ip'):
            return None

        src_port = dst_port = 0
        protocol = 'OTHER'

        if hasattr(pkt, 'tcp'):
            protocol = 'TCP'
            src_port = int(pkt.tcp.srcport)
            dst_port = int(pkt.tcp.dstport)
        elif hasattr(pkt, 'udp'):
            protocol = 'UDP'
            src_port = int(pkt.udp.srcport)
            dst_port = int(pkt.udp.dstport)

        return {
            'timestamp': float(pkt.sniff_timestamp),
            'src_ip': pkt.ip.src,
            'dst_ip': pkt.ip.dst,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol,
            'length': int(pkt.length),
        }
    except:
        return None


def start_live_capture(interface=None):
    global _capture_thread

    with _lock:
        if _capture_thread and _capture_thread.is_alive():
            print("[LIVE] Захват уже работает!")
            return

        def capture_loop():
            print(f"[LIVE] Запуск захвата: {interface or 'auto'}")

            # FIX для Windows: создаём event loop вручную
            try:
                asyncio.set_event_loop(asyncio.new_event_loop())
            except Exception as e:
                print("[LIVE] Ошибка создания event loop:", e)

            try:
                cap = pyshark.LiveCapture(
                    interface=interface,
                    use_json=True,
                    include_raw=True
                )

                for pkt in cap.sniff_continuously(packet_count=None):
                    data = packet_handler(pkt)
                    if data:
                        try:
                            packet_queue.put_nowait(data)
                        except queue.Full:
                            pass

            except Exception as e:
                print("[LIVE] Ошибка захвата:", e)
            finally:
                print("[LIVE] Захват остановлен")

        _capture_thread = threading.Thread(target=capture_loop, daemon=True)
        _capture_thread.start()
        time.sleep(1)
        print("[LIVE] Захват запущен!")


def stop_live_capture():
    global _capture_thread
    with _lock:
        if _capture_thread and _capture_thread.is_alive():
            print("[LIVE] Остановка...")
            _capture_thread = None
