from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


APP_TITLE = "Excel AI 半自动标注工具"
APP_FOLDER = "ExcelAILabeler"
LOADING_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: #f7f8fa;
  color: #20242a;
  font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
}
.loader { width: min(360px, calc(100vw - 48px)); text-align: center; }
.mark {
  width: 58px;
  height: 58px;
  margin: 0 auto 22px;
  border-radius: 12px;
  background: #167d6b;
  display: grid;
  place-items: center;
  color: white;
  font-size: 20px;
  font-weight: 700;
}
h1 { margin: 0 0 10px; font-size: 22px; font-weight: 650; letter-spacing: 0; }
p { margin: 0 0 24px; color: #667085; font-size: 14px; }
.track { height: 4px; overflow: hidden; background: #e4e7ec; border-radius: 2px; }
.bar {
  width: 42%;
  height: 100%;
  background: #167d6b;
  border-radius: 2px;
  animation: loading 1s ease-in-out infinite alternate;
}
@keyframes loading { from { transform: translateX(-20%); } to { transform: translateX(158%); } }
</style>
</head>
<body>
  <main class="loader">
    <div class="mark">AI</div>
    <h1>Excel AI 标注</h1>
    <p>正在启动本地工作区...</p>
    <div class="track"><div class="bar"></div></div>
  </main>
</body>
</html>
"""


def message(text: str, error: bool = False) -> None:
    flags = 0x10 if error else 0x40
    ctypes.windll.user32.MessageBoxW(None, text, APP_TITLE, flags)


def project_dir() -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        candidates = (executable_dir, executable_dir.parent)
    else:
        candidates = (Path(__file__).resolve().parent,)
    for candidate in candidates:
        if (candidate / "app.py").is_file() and (candidate / ".venv").is_dir():
            return candidate
    return candidates[0]


def runtime_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    path = Path(root) / APP_FOLDER
    path.mkdir(parents=True, exist_ok=True)
    return path


def acquire_single_instance_lock():
    import msvcrt

    lock_path = runtime_dir() / "app.lock"
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return handle
    except OSError:
        handle.close()
        return None


def choose_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(port: int, process: subprocess.Popen, timeout: float = 40.0) -> bool:
    health = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health, timeout=1) as response:
                if response.read().decode("utf-8", errors="ignore").strip() == "ok":
                    return True
        except OSError:
            time.sleep(0.25)
    return False


def write_runtime_state(port: int, server_pid: int) -> None:
    state = {"port": port, "server_pid": server_pid, "started_at": time.time()}
    (runtime_dir() / "runtime.json").write_text(json.dumps(state), encoding="utf-8")


def stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=False,
    )


def main() -> int:
    lock = acquire_single_instance_lock()
    if lock is None:
        message("程序已经在运行。请查看任务栏中的应用窗口。")
        return 0

    root = project_dir()
    app_file = root / "app.py"
    pythonw = root / ".venv" / "Scripts" / "pythonw.exe"
    if not app_file.is_file() or not pythonw.is_file():
        message("程序文件不完整，请重新安装。", error=True)
        return 1

    port = choose_port()
    logs = runtime_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_file = (logs / "server.log").open("w", encoding="utf-8")
    stderr_file = (logs / "server-error.log").open("w", encoding="utf-8")
    creation_flags = subprocess.CREATE_NO_WINDOW
    server = subprocess.Popen(
        [
            str(pythonw),
            "-m",
            "streamlit",
            "run",
            str(app_file),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=root,
        stdout=stdout_file,
        stderr=stderr_file,
        creationflags=creation_flags,
    )
    try:
        write_runtime_state(port, server.pid)
        import webview

        profile = runtime_dir() / "WebViewProfile"
        profile.mkdir(parents=True, exist_ok=True)
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
        window = webview.create_window(
            APP_TITLE,
            html=LOADING_HTML,
            width=1280,
            height=850,
            min_size=(900, 650),
            resizable=True,
            background_color="#F7F8FA",
            text_select=True,
        )
        icon = root / "app_icon.ico"

        startup_failed: list[bool] = []

        def load_application() -> None:
            if wait_for_server(port, server):
                window.load_url(f"http://127.0.0.1:{port}")
                return
            startup_failed.append(True)
            window.load_html(
                "<html><body style='font-family:Segoe UI;padding:40px'>"
                "<h2>本地服务启动失败</h2>"
                f"<p>请检查日志：{logs}</p></body></html>"
            )

        webview.start(
            load_application,
            gui="edgechromium",
            debug=False,
            private_mode=False,
            storage_path=str(profile),
            icon=str(icon) if icon.is_file() else None,
        )
        return 1 if startup_failed else 0
    except Exception as exc:
        message(f"程序启动失败：{exc}", error=True)
        return 1
    finally:
        stop_process_tree(server)
        stdout_file.close()
        stderr_file.close()
        (runtime_dir() / "runtime.json").unlink(missing_ok=True)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
