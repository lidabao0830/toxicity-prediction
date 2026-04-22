import os
import sys
import time
import socket
import threading
import subprocess
import webview


def find_free_port(start=8501, end=8999):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("未找到可用端口")


def wait_for_server(host, port, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def run_streamlit(port):
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(cmd)


if __name__ == "__main__":
    port = find_free_port()
    process = run_streamlit(port)

    ok = wait_for_server("127.0.0.1", port, timeout=40)
    if not ok:
        process.terminate()
        raise RuntimeError("Streamlit 服务启动失败")

    url = f"http://127.0.0.1:{port}"

    window = webview.create_window(
        "毒性预测系统",
        url=url,
        width=1400,
        height=900,
        resizable=True
    )

    try:
        webview.start()
    finally:
        process.terminate()
