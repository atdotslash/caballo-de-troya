"""Resolución de rutas en tiempo de ejecución (compatible con Python y PyInstaller .exe)."""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """Devuelve True cuando la aplicación se ejecuta empaquetada como .exe (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def get_runtime_dir() -> Path:
    """Carpeta editable externa: junto a `CaballoDeTroya.exe` o raíz del proyecto en desarrollo."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_bundle_dir() -> Path:
    """Carpeta interna de recursos empaquetados (`sys._MEIPASS`) o raíz del proyecto."""
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parent.parent
