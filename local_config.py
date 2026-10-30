from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path
from typing import Any


APP_FOLDER = "ExcelAILabeler"
CONFIG_FILE = "settings.json"
CONFIG_VERSION = 1


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def app_data_dir() -> Path:
    root = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(root) / APP_FOLDER


def config_path() -> Path:
    return app_data_dir() / CONFIG_FILE


def _blob_from_bytes(data: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
    buffer = ctypes.create_string_buffer(data)
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    return blob, buffer


def protect_secret(secret: str) -> str:
    if not secret:
        return ""
    if os.name != "nt":
        raise OSError("API Key 安全存储仅支持 Windows")

    raw = secret.encode("utf-8")
    input_blob, _buffer = _blob_from_bytes(raw)
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "Excel AI Labeler API Key",
        None,
        None,
        None,
        0,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    try:
        protected = ctypes.string_at(output_blob.pbData, output_blob.cbData)
        return base64.b64encode(protected).decode("ascii")
    finally:
        kernel32.LocalFree(output_blob.pbData)


def unprotect_secret(protected: str) -> str:
    if not protected:
        return ""
    if os.name != "nt":
        return ""
    try:
        raw = base64.b64decode(protected.encode("ascii"), validate=True)
        input_blob, _buffer = _blob_from_bytes(raw)
        output_blob = _DataBlob()
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        if not crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(output_blob),
        ):
            return ""
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData).decode("utf-8")
        finally:
            kernel32.LocalFree(output_blob.pbData)
    except (ValueError, UnicodeDecodeError, OSError):
        return ""


def load_settings(path: Path | None = None) -> dict[str, Any]:
    target = path or config_path()
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        settings = payload.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
        settings["remember_api_key"] = bool(payload.get("remember_api_key", False))
        if settings["remember_api_key"]:
            settings["api_key"] = unprotect_secret(str(payload.get("api_key_protected", "")))
        return settings
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(
    settings: dict[str, Any],
    api_key: str,
    remember_api_key: bool,
    path: Path | None = None,
) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CONFIG_VERSION,
        "remember_api_key": bool(remember_api_key),
        "api_key_protected": protect_secret(api_key) if remember_api_key and api_key else "",
        "settings": settings,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return target


def clear_settings(path: Path | None = None) -> bool:
    target = path or config_path()
    try:
        target.unlink(missing_ok=True)
        return True
    except OSError:
        return False
