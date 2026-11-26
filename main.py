"""Punto de entrada principal de Caballo de Troya (Windows 10 / .exe compatible)."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from src.paths import get_bundle_dir, get_runtime_dir, is_frozen

RUNTIME_DIR = get_runtime_dir()
if not is_frozen() and str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from src.gui import ChessRobotApp  # noqa: E402


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_runtime_config() -> Path:
    """Asegura que `config.json` exista en el directorio externo junto al ejecutable."""
    config_file = RUNTIME_DIR / "config.json"
    if not config_file.exists():
        bundled = get_bundle_dir() / "config.json"
        if bundled.exists() and bundled != config_file:
            try:
                shutil.copy2(bundled, config_file)
            except OSError:
                pass
    return config_file


def main() -> None:
    configure_logging()
    config_file = ensure_runtime_config()
    app = ChessRobotApp(config_path=str(config_file))
    app.mainloop()


if __name__ == "__main__":
    main()
