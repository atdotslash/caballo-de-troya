"""Gestión segura y persistente del archivo de configuración config.json."""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "serial": {
        "port": "",
        "baudrate": 115200,
        "timeout": 1.0,
        "status_poll_interval_s": 0.25,
        "command_timeout_s": 30.0,
    },
    "motion": {
        "z_safe_mm": 45.0,
        "z_pick_mm": 5.0,
        "z_place_mm": 7.0,
        "default_jog_feed_mm_min": 1800.0,
        "plunge_feed_mm_min": 900.0,
        "travel_mode": "G0",
        "standby_position": {"x": 10.0, "y": 10.0, "z": 45.0},
        "capture_zone": {"x": 290.0, "y": 20.0, "z_drop": 12.0, "step_y": 28.0},
    },
    "gripper": {
        "open_pwm": 65,
        "close_pwm": 155,
        "min_pwm": 0,
        "max_pwm": 255,
        "dwell_seconds": 0.45,
        "disable_pwm_after_move": False,
    },
    "chess_clock": {
        "enabled": True,
        "x": 280.0,
        "y": 140.0,
        "z_press": 12.0,
        "press_dwell_seconds": 0.35,
    },
    "grbl_presets": {
        "active_preset": "normal",
        "presets": {
            "suave": {
                "label": "Suave (Calibración)",
                "$110": 1500.0,
                "$111": 1500.0,
                "$112": 800.0,
                "$120": 80.0,
                "$121": 80.0,
                "$122": 50.0,
            },
            "normal": {
                "label": "Normal (Partida)",
                "$110": 3000.0,
                "$111": 3000.0,
                "$112": 1500.0,
                "$120": 200.0,
                "$121": 200.0,
                "$122": 120.0,
            },
            "rapido": {
                "label": "Rápido (Blitz)",
                "$110": 5000.0,
                "$111": 5000.0,
                "$112": 2200.0,
                "$120": 400.0,
                "$121": 400.0,
                "$122": 200.0,
            },
        },
    },
    "vision": {
        "camera_index": 0,
        "frame_width": 1280,
        "frame_height": 720,
        "aruco_dictionary": "DICT_5X5_50",
        "calibration": {
            "calibrated": False,
            "corner_order": [
                "A8_superior_izquierda",
                "H8_superior_derecha",
                "H1_inferior_derecha",
                "A1_inferior_izquierda",
            ],
            "pixel_corners": [],
            "machine_corners_mm": [
                [20.0, 260.0],
                [260.0, 260.0],
                [260.0, 20.0],
                [20.0, 20.0],
            ],
        },
    },
    "audio": {
        "enabled": True,
        "use_tts": False,
        "preferred_locale": "es-AR",
        "rate": 175,
        "volume": 1.0,
        "humor_enabled": True,
        "humor_timer_enabled": True,
        "delay_warning_seconds": 30,
        "sounds_dir": "",
        "phrases_path": "assets/frases_humor.json",
    },
    "chess": {
        "engine_path": "",
        "engine_think_time_s": 1.0,
        "robot_color": "black",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Combina recursivamente un diccionario de usuario sobre los valores por defecto."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class ConfigManager:
    """Administra la lectura y escritura atómica y thread-safe de config.json."""

    def __init__(self, config_path: str | Path = "config.json") -> None:
        self.path = Path(config_path).resolve()
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        self.load()

    @property
    def data(self) -> Dict[str, Any]:
        with self._lock:
            return self._data

    def load(self) -> Dict[str, Any]:
        """Carga config.json desde disco; si no existe, lo crea con defaults."""
        with self._lock:
            if not self.path.exists():
                self._data = copy.deepcopy(DEFAULT_CONFIG)
                self.save()
                return self._data

            try:
                with self.path.open("r", encoding="utf-8") as fh:
                    user_data = json.load(fh)
                if not isinstance(user_data, dict):
                    raise ValueError("La raíz de config.json debe ser un objeto JSON.")
                self._data = _deep_merge(DEFAULT_CONFIG, user_data)
            except (OSError, json.JSONDecodeError, ValueError):
                # Respalda el archivo dañado y regenera configuración limpia
                backup_path = self.path.with_suffix(".json.bak")
                try:
                    os.replace(self.path, backup_path)
                except OSError:
                    pass
                self._data = copy.deepcopy(DEFAULT_CONFIG)
                self.save()

            return self._data

    def save(self) -> None:
        """Guarda la configuración de forma atómica para evitar corrupción."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp_path, self.path)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Accede a claves anidadas con seguridad."""
        with self._lock:
            current: Any = self._data
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            return copy.deepcopy(current)

    def set(self, *keys: str, value: Any, autosave: bool = True) -> None:
        """Actualiza una clave anidada y opcionalmente persiste en disco."""
        if not keys:
            raise ValueError("Debe especificarse al menos una clave.")
        with self._lock:
            current = self._data
            for key in keys[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = copy.deepcopy(value)
            if autosave:
                self.save()
