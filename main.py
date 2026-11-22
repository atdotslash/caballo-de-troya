"""Punto de entrada principal de Caballo de Troya (Windows 10)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Asegura que el directorio raíz esté en sys.path al ejecutar directamente
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gui import ChessRobotApp  # noqa: E402


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    configure_logging()
    config_file = ROOT_DIR / "config.json"
    app = ChessRobotApp(config_path=str(config_file))
    app.mainloop()


if __name__ == "__main__":
    main()
