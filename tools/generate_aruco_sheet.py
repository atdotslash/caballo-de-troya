"""Genera una planilla imprimible con los 32 marcadores ArUco (DICT_5X5_50) para las piezas.

Uso:
    .venv\\Scripts\\python.exe tools\\generate_aruco_sheet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vision import ARUCO_PIECE_MAP  # noqa: E402


def generate_sheet(output_path: Path, marker_size_px: int = 200) -> Path:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

    cols = 8
    rows = 4
    pad = 40
    label_h = 36
    cell_w = marker_size_px + pad
    cell_h = marker_size_px + pad + label_h

    sheet = np.full((rows * cell_h + pad, cols * cell_w + pad, 3), 255, dtype=np.uint8)

    for marker_id in range(32):
        row = marker_id // cols
        col = marker_id % cols
        x0 = pad + col * cell_w
        y0 = pad + row * cell_h

        marker_gray = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
        marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
        sheet[y0 : y0 + marker_size_px, x0 : x0 + marker_size_px] = marker_bgr

        # Borde de recorte suave
        cv2.rectangle(
            sheet,
            (x0 - 8, y0 - 8),
            (x0 + marker_size_px + 8, y0 + marker_size_px + label_h),
            (190, 190, 190),
            1,
        )

        spec = ARUCO_PIECE_MAP[marker_id]
        caption = f"#{marker_id:02d} {spec.short_es}"
        cv2.putText(
            sheet,
            caption,
            (x0 + 4, y0 + marker_size_px + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return output_path


if __name__ == "__main__":
    out = ROOT_DIR / "assets" / "aruco_32_piezas_DICT_5X5_50.png"
    saved = generate_sheet(out)
    print(f"Planilla ArUco generada correctamente en: {saved}")
