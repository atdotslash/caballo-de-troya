"""Módulo de Visión Artificial para Caballo de Troya.

Características:
- Detección de 32 marcadores ArUco (cv2.aruco.DICT_5X5_50) compatible con OpenCV 4.10+ / 5.x.
- Mapeo intercambiable de IDs 0-15 (Blancas) y 16-31 (Negras) a tipo y color de pieza.
- Calibración de tablero de 4 esquinas mediante homografía (píxeles <-> tablero 8x8 <-> mm máquina).
- Superposición visual en vivo de cajas ArUco, nombres en español y grilla 8x8 con coordenadas algebraicas.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FILES = "abcdefgh"
RANKS = "12345678"

CORNER_INSTRUCTIONS_ES: List[str] = [
    "1/4: Hacé clic en la esquina exterior SUPERIOR IZQUIERDA (A8)",
    "2/4: Hacé clic en la esquina exterior SUPERIOR DERECHA (H8)",
    "3/4: Hacé clic en la esquina exterior INFERIOR DERECHA (H1)",
    "4/4: Hacé clic en la esquina exterior INFERIOR IZQUIERDA (A1)",
]


@dataclass(frozen=True)
class PieceSpec:
    """Especificación canónica de una pieza de ajedrez (independiente del ID exacto)."""

    color: str  # "white" | "black"
    piece_type: str  # "R", "N", "B", "Q", "K", "P"
    fen_symbol: str  # "R","N","B","Q","K","P" (blancas) o "r","n","b","q","k","p" (negras)
    name_es: str  # Nombre legible en Español (Argentina)
    short_es: str  # Etiqueta corta para overlay


def _build_aruco_piece_map() -> Dict[int, PieceSpec]:
    """Construye la tabla oficial de 32 marcadores ArUco (IDs 0-31)."""
    mapping: Dict[int, PieceSpec] = {}

    white_specs = {
        0: PieceSpec("white", "R", "R", "Torre Blanca", "TB"),
        7: PieceSpec("white", "R", "R", "Torre Blanca", "TB"),
        1: PieceSpec("white", "N", "N", "Caballo Blanco", "CB"),
        6: PieceSpec("white", "N", "N", "Caballo Blanco", "CB"),
        2: PieceSpec("white", "B", "B", "Alfil Blanco", "AB"),
        5: PieceSpec("white", "B", "B", "Alfil Blanco", "AB"),
        3: PieceSpec("white", "Q", "Q", "Dama Blanca", "DB"),
        4: PieceSpec("white", "K", "K", "Rey Blanco", "RB"),
    }
    for pid in range(8, 16):
        white_specs[pid] = PieceSpec("white", "P", "P", "Peón Blanco", "PB")

    black_specs = {
        16: PieceSpec("black", "R", "r", "Torre Negra", "TN"),
        23: PieceSpec("black", "R", "r", "Torre Negra", "TN"),
        17: PieceSpec("black", "N", "n", "Caballo Negro", "CN"),
        22: PieceSpec("black", "N", "n", "Caballo Negro", "CN"),
        18: PieceSpec("black", "B", "b", "Alfil Negro", "AN"),
        21: PieceSpec("black", "B", "b", "Alfil Negro", "AN"),
        19: PieceSpec("black", "Q", "q", "Dama Negra", "DN"),
        20: PieceSpec("black", "K", "k", "Rey Negro", "RN"),
    }
    for pid in range(24, 32):
        black_specs[pid] = PieceSpec("black", "P", "p", "Peón Negro", "PN")

    mapping.update(white_specs)
    mapping.update(black_specs)
    return mapping


ARUCO_PIECE_MAP: Dict[int, PieceSpec] = _build_aruco_piece_map()


@dataclass
class DetectedPiece:
    """Representa un marcador ArUco detectado en un fotograma."""

    marker_id: int
    piece: Optional[PieceSpec]
    center_px: Tuple[float, float]
    corners_px: np.ndarray  # shape (4, 2)
    square: Optional[str] = None  # Ej: "e4" si está calibrado y dentro del tablero
    machine_mm: Optional[Tuple[float, float]] = None  # (X_mm, Y_mm)


class BoardCalibrator:
    """Gestiona la homografía de 4 esquinas entre cámara (px), tablero (8x8) y máquina (mm)."""

    # Orden de las 4 esquinas exteriores del área de juego 8x8:
    # A8 (top-left), H8 (top-right), H1 (bottom-right), A1 (bottom-left)
    BOARD_CORNERS_UNIT = np.array(
        [
            [0.0, 0.0],
            [8.0, 0.0],
            [8.0, 8.0],
            [0.0, 8.0],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        pixel_corners: Optional[Sequence[Sequence[float]]] = None,
        machine_corners_mm: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.pixel_corners: Optional[np.ndarray] = None
        self.machine_corners_mm: np.ndarray = np.array(
            machine_corners_mm
            if machine_corners_mm and len(machine_corners_mm) == 4
            else [
                [20.0, 260.0],
                [260.0, 260.0],
                [260.0, 20.0],
                [20.0, 20.0],
            ],
            dtype=np.float32,
        )

        self._h_px_to_board: Optional[np.ndarray] = None
        self._h_board_to_px: Optional[np.ndarray] = None
        self._h_px_to_mm: Optional[np.ndarray] = None
        self._h_board_to_mm: Optional[np.ndarray] = None

        self._compute_board_to_mm()
        if pixel_corners and len(pixel_corners) == 4:
            self.set_pixel_corners(pixel_corners)

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return self._h_px_to_board is not None and self._h_px_to_mm is not None

    def _compute_board_to_mm(self) -> None:
        with self._lock:
            self._h_board_to_mm = cv2.getPerspectiveTransform(
                self.BOARD_CORNERS_UNIT, self.machine_corners_mm
            )

    def set_machine_corners_mm(self, corners_mm: Sequence[Sequence[float]]) -> None:
        if len(corners_mm) != 4:
            raise ValueError("Se requieren exactamente 4 esquinas en milímetros.")
        with self._lock:
            self.machine_corners_mm = np.array(corners_mm, dtype=np.float32)
            self._compute_board_to_mm()
            if self.pixel_corners is not None:
                self._h_px_to_mm = cv2.getPerspectiveTransform(
                    self.pixel_corners, self.machine_corners_mm
                )

    def set_pixel_corners(self, corners_px: Sequence[Sequence[float]]) -> None:
        if len(corners_px) != 4:
            raise ValueError("Se requieren exactamente 4 esquinas en píxeles.")
        with self._lock:
            pts = np.array(corners_px, dtype=np.float32)
            self.pixel_corners = pts
            self._h_px_to_board = cv2.getPerspectiveTransform(pts, self.BOARD_CORNERS_UNIT)
            self._h_board_to_px = cv2.getPerspectiveTransform(self.BOARD_CORNERS_UNIT, pts)
            self._h_px_to_mm = cv2.getPerspectiveTransform(pts, self.machine_corners_mm)

    def clear_pixel_calibration(self) -> None:
        with self._lock:
            self.pixel_corners = None
            self._h_px_to_board = None
            self._h_board_to_px = None
            self._h_px_to_mm = None

    def pixel_to_board(self, x_px: float, y_px: float) -> Optional[Tuple[float, float]]:
        """Transforma un píxel (x, y) a coordenadas de tablero (u, v) en [0..8] x [0..8]."""
        with self._lock:
            if self._h_px_to_board is None:
                return None
            src = np.array([[[x_px, y_px]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self._h_px_to_board)[0, 0]
            return float(dst[0]), float(dst[1])

    def pixel_to_square(self, x_px: float, y_px: float) -> Optional[str]:
        """Devuelve la casilla algebraica ('a1'..'h8') si el píxel cae dentro del tablero."""
        uv = self.pixel_to_board(x_px, y_px)
        if uv is None:
            return None
        u, v = uv
        if not (0.0 <= u < 8.0 and 0.0 <= v < 8.0):
            return None
        file_idx = int(np.floor(u))
        rank_idx = 7 - int(np.floor(v))  # v=0 es fila 8 (arriba), v=8 es fila 1 (abajo)
        return f"{FILES[file_idx]}{RANKS[rank_idx]}"

    def pixel_to_machine_mm(self, x_px: float, y_px: float) -> Optional[Tuple[float, float]]:
        """Transforma un píxel de cámara directamente a coordenadas reales de máquina (X, Y) en mm."""
        with self._lock:
            if self._h_px_to_mm is None:
                return None
            src = np.array([[[x_px, y_px]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self._h_px_to_mm)[0, 0]
            return float(dst[0]), float(dst[1])

    def square_to_machine_mm(self, square_name: str) -> Tuple[float, float]:
        """Devuelve el centro físico (X_mm, Y_mm) de una casilla algebraica ('e4')."""
        sq = square_name.strip().lower()
        if len(sq) != 2 or sq[0] not in FILES or sq[1] not in RANKS:
            raise ValueError(f"Casilla algebraica inválida: '{square_name}'")
        file_idx = FILES.index(sq[0])
        rank_idx = RANKS.index(sq[1])
        u = file_idx + 0.5
        v = (7 - rank_idx) + 0.5
        with self._lock:
            assert self._h_board_to_mm is not None
            src = np.array([[[u, v]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self._h_board_to_mm)[0, 0]
            return float(dst[0]), float(dst[1])

    def board_points_to_pixels(self, points_uv: np.ndarray) -> Optional[np.ndarray]:
        """Proyecta puntos del plano del tablero (N, 2) de vuelta a píxeles para dibujar overlays."""
        with self._lock:
            if self._h_board_to_px is None:
                return None
            pts = np.asarray(points_uv, dtype=np.float32).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(pts, self._h_board_to_px)
            return projected.reshape(-1, 2)


class VisionSystem:
    """Captura de cámara cenital en hilo dedicado + detección ArUco + overlay 8x8."""

    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        calibrator: Optional[BoardCalibrator] = None,
    ) -> None:
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.calibrator = calibrator or BoardCalibrator()

        # API moderna OpenCV 4.10+ / 5.x
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self._aruco_params = cv2.aruco.DetectorParameters()
        # Ajuste suave para mejorar detección en marcadores pequeños de piezas
        self._aruco_params.adaptiveThreshWinSizeMin = 3
        self._aruco_params.adaptiveThreshWinSizeMax = 23
        self._aruco_params.adaptiveThreshWinSizeStep = 10
        self._aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_detections: List[DetectedPiece] = []
        self._fps: float = 0.0
        self._last_error: str = ""

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def last_error(self) -> str:
        return self._last_error

    def start(self, camera_index: Optional[int] = None) -> bool:
        """Inicia la captura de video en un hilo en segundo plano."""
        if camera_index is not None:
            self.camera_index = int(camera_index)

        self.stop()
        self._last_error = ""

        # En Windows 10, CAP_DSHOW suele iniciar más rápido y sin congelamientos
        cap: Optional[cv2.VideoCapture] = None
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.camera_index)
        else:
            cap = cv2.VideoCapture(self.camera_index)

        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            self._last_error = f"No se pudo abrir la cámara índice {self.camera_index}."
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="VisionCaptureThread", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Detiene la captura y libera el dispositivo de cámara."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        with self._lock:
            self._latest_raw_frame = None
            self._latest_detections = []
            self._fps = 0.0

    def _capture_loop(self) -> None:
        frame_count = 0
        t0 = time.perf_counter()

        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            detections = self.detect_pieces(frame)

            with self._lock:
                self._latest_raw_frame = frame
                self._latest_detections = detections

            frame_count += 1
            now = time.perf_counter()
            if now - t0 >= 1.0:
                self._fps = frame_count / (now - t0)
                frame_count = 0
                t0 = now

    def detect_pieces(self, frame_bgr: np.ndarray) -> List[DetectedPiece]:
        """Detecta marcadores ArUco DICT_5X5_50 y calcula casilla + coordenadas en mm."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)
        results: List[DetectedPiece] = []

        if ids is None or len(ids) == 0:
            return results

        flat_ids = np.asarray(ids).reshape(-1)
        for marker_corners, marker_id_val in zip(corners_list, flat_ids):
            marker_id = int(marker_id_val)
            pts = marker_corners.reshape(4, 2)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

            piece_spec = ARUCO_PIECE_MAP.get(marker_id)
            square = self.calibrator.pixel_to_square(cx, cy)
            machine_mm = self.calibrator.pixel_to_machine_mm(cx, cy)

            results.append(
                DetectedPiece(
                    marker_id=marker_id,
                    piece=piece_spec,
                    center_px=(cx, cy),
                    corners_px=pts,
                    square=square,
                    machine_mm=machine_mm,
                )
            )
        return results

    def get_latest_state(self) -> Tuple[Optional[np.ndarray], List[DetectedPiece]]:
        """Devuelve una copia segura del último frame capturado y sus detecciones."""
        with self._lock:
            frame = None if self._latest_raw_frame is None else self._latest_raw_frame.copy()
            detections = list(self._latest_detections)
            return frame, detections

    def get_board_occupancy_map(self) -> Dict[str, str]:
        """Devuelve un diccionario {casilla: simbolo_FEN} con las piezas detectadas sobre el tablero.

        Las piezas del mismo tipo y color son completamente intercambiables porque se
        representan mediante su símbolo FEN canónico ('P','R','N','B','Q','K','p','r','n','b','q','k').
        """
        _, detections = self.get_latest_state()
        occupancy: Dict[str, str] = {}
        for det in detections:
            if det.square and det.piece is not None:
                occupancy[det.square] = det.piece.fen_symbol
        return occupancy

    def render_overlay(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[DetectedPiece],
        show_grid: bool = True,
        show_aruco: bool = True,
        pending_calibration_points: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Dibuja la grilla 8x8 calibrada, las cajas ArUco y los puntos de calibración en curso."""
        canvas = frame_bgr.copy()

        # 1. Grilla 8x8 proyectada con homografía
        if show_grid and self.calibrator.is_calibrated:
            self._draw_board_grid(canvas)

        # 2. Puntos de calibración temporales si el usuario está calibrando
        if pending_calibration_points:
            labels = ["A8", "H8", "H1", "A1"]
            for idx, (px, py) in enumerate(pending_calibration_points):
                pt = (int(round(px)), int(round(py)))
                cv2.circle(canvas, pt, 8, (0, 215, 255), -1)
                cv2.circle(canvas, pt, 12, (255, 255, 255), 2)
                lbl = labels[idx] if idx < len(labels) else str(idx + 1)
                cv2.putText(
                    canvas,
                    lbl,
                    (pt[0] + 12, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )
            if len(pending_calibration_points) >= 2:
                poly = np.array(pending_calibration_points, dtype=np.int32)
                cv2.polylines(canvas, [poly], False, (0, 215, 255), 2, cv2.LINE_AA)

        # 3. Marcadores ArUco detectados
        if show_aruco:
            for det in detections:
                pts = det.corners_px.astype(np.int32)
                cx, cy = int(round(det.center_px[0])), int(round(det.center_px[1]))

                if det.piece is None:
                    box_color = (0, 140, 255)  # Naranja: ID fuera de 0-31
                    label = f"ID {det.marker_id} (?)"
                elif det.piece.color == "white":
                    box_color = (80, 235, 120)  # Verde brillante: Blancas
                    label = f"{det.piece.name_es} (#{det.marker_id})"
                else:
                    box_color = (255, 165, 60)  # Celeste/Azul: Negras
                    label = f"{det.piece.name_es} (#{det.marker_id})"

                if det.square:
                    label = f"{label} [{det.square.upper()}]"

                cv2.polylines(canvas, [pts], True, box_color, 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

                # Etiqueta con fondo semioscuro para alta legibilidad
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.48
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
                tx = max(4, cx - tw // 2)
                ty = max(th + 8, int(np.min(pts[:, 1])) - 8)
                cv2.rectangle(
                    canvas,
                    (tx - 4, ty - th - 4),
                    (tx + tw + 4, ty + 4),
                    (20, 20, 24),
                    -1,
                )
                cv2.putText(
                    canvas,
                    label,
                    (tx, ty),
                    font,
                    scale,
                    box_color,
                    thickness,
                    cv2.LINE_AA,
                )

        return canvas

    def _draw_board_grid(self, canvas: np.ndarray) -> None:
        """Dibuja la grilla 8x8 y las coordenadas algebraicas proyectadas en perspectiva."""
        # Líneas verticales y horizontales
        for i in range(9):
            v_pts = self.calibrator.board_points_to_pixels(np.array([[i, 0], [i, 8]], dtype=np.float32))
            h_pts = self.calibrator.board_points_to_pixels(np.array([[0, i], [8, i]], dtype=np.float32))
            color = (0, 255, 255) if i in (0, 8) else (60, 200, 200)
            thickness = 2 if i in (0, 8) else 1
            if v_pts is not None:
                p1 = tuple(np.round(v_pts[0]).astype(int))
                p2 = tuple(np.round(v_pts[1]).astype(int))
                cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
            if h_pts is not None:
                p1 = tuple(np.round(h_pts[0]).astype(int))
                p2 = tuple(np.round(h_pts[1]).astype(int))
                cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

        # Etiquetas de columnas (a-h) y filas (1-8)
        file_centers = np.array([[i + 0.5, 8.28] for i in range(8)], dtype=np.float32)
        rank_centers = np.array([[-0.32, (7 - r) + 0.55] for r in range(8)], dtype=np.float32)
        f_px = self.calibrator.board_points_to_pixels(file_centers)
        r_px = self.calibrator.board_points_to_pixels(rank_centers)

        if f_px is not None:
            for idx, pt in enumerate(f_px):
                cv2.putText(
                    canvas,
                    FILES[idx].upper(),
                    (int(pt[0]) - 6, int(pt[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        if r_px is not None:
            for idx, pt in enumerate(r_px):
                cv2.putText(
                    canvas,
                    RANKS[idx],
                    (int(pt[0]) - 6, int(pt[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
