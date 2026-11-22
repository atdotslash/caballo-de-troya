"""Motor lógico de ajedrez basado en python-chess.

Funciones principales:
- Mantenimiento y validación estricta del estado legal del tablero (`chess.Board`).
- Detección de jugadas humanas comparando el mapa ArUco observado (piezas intercambiables
  por tipo y color) contra todas las jugadas legales posibles (incluyendo capturas, enroques,
  captura al paso y promociones).
- Integración opcional con un motor UCI local (por ejemplo, Stockfish) o ingreso manual de jugadas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine

logger = logging.getLogger(__name__)


@dataclass
class MoveOutcome:
    """Resultado estructurado de validar o ejecutar una jugada."""

    valid: bool
    message_es: str
    move_uci: Optional[str] = None
    is_capture: bool = False
    is_castling: bool = False
    rook_castling_move: Optional[Tuple[str, str]] = None  # (origen_torre, destino_torre)
    audio_events: Tuple[str, ...] = ()


class ChessGameEngine:
    """Encapsula `python-chess` y opcionalmente un motor UCI externo (Stockfish)."""

    def __init__(self, engine_path: str = "", think_time_s: float = 1.0) -> None:
        self.board = chess.Board()
        self.engine_path = engine_path
        self.think_time_s = think_time_s
        self._uci_engine: Optional[chess.engine.SimpleEngine] = None

    def reset_game(self) -> None:
        self.board.reset()

    def get_board_occupancy(self, board: Optional[chess.Board] = None) -> Dict[str, str]:
        """Convierte un `chess.Board` al mismo formato `{casilla: simbolo_FEN}` que devuelve VisionSystem."""
        target = board or self.board
        occupancy: Dict[str, str] = {}
        for sq, piece in target.piece_map().items():
            occupancy[chess.square_name(sq)] = piece.symbol()
        return occupancy

    def get_turn_event_key(self) -> str:
        return "white_turn" if self.board.turn == chess.WHITE else "black_turn"

    def get_status_text_es(self) -> str:
        if self.board.is_checkmate():
            ganador = "Negras" if self.board.turn == chess.WHITE else "Blancas"
            return f"Jaque Mate — Ganan las {ganador}"
        if self.board.is_stalemate():
            return "Tablas por rey ahogado"
        if self.board.is_insufficient_material() or self.board.is_fifty_moves() or self.board.is_repetition():
            return "Tablas"
        turno = "Blancas" if self.board.turn == chess.WHITE else "Negras"
        if self.board.is_check():
            return f"¡Jaque! Turno de las {turno}"
        return f"Turno de las {turno}"

    def _compute_castling_rook_move(self, move: chess.Move) -> Optional[Tuple[str, str]]:
        if not self.board.is_castling(move):
            return None
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        mapping = {
            ("e1", "g1"): ("h1", "f1"),
            ("e1", "c1"): ("a1", "d1"),
            ("e8", "g8"): ("h8", "f8"),
            ("e8", "c8"): ("a8", "d8"),
        }
        return mapping.get((from_sq, to_sq))

    def _collect_post_move_audio_events(self) -> Tuple[str, ...]:
        events: List[str] = []
        if self.board.is_checkmate():
            events.append("checkmate")
        elif self.board.is_stalemate():
            events.append("stalemate")
        elif (
            self.board.is_insufficient_material()
            or self.board.is_fifty_moves()
            or self.board.is_repetition()
        ):
            events.append("draw")
        else:
            if self.board.is_check():
                events.append("check")
            events.append(self.get_turn_event_key())
        return tuple(events)

    def apply_move_str(self, move_text: str) -> MoveOutcome:
        """Valida y aplica una jugada en notación UCI ('e2e4') o SAN ('Cf3'/'Nf3')."""
        clean = move_text.strip()
        if not clean:
            return MoveOutcome(False, "Ingresá una jugada válida (ej. e2e4).", audio_events=("invalid_move",))

        move: Optional[chess.Move] = None
        try:
            candidate = chess.Move.from_uci(clean.lower())
            if candidate in self.board.legal_moves:
                move = candidate
        except ValueError:
            pass

        if move is None:
            try:
                move = self.board.parse_san(clean)
            except ValueError:
                return MoveOutcome(
                    False,
                    f"Movimiento inválido o ilegal: '{clean}'.",
                    audio_events=("invalid_move",),
                )

        return self._commit_move(move)

    def detect_human_move_from_vision(self, observed_occupancy: Dict[str, str]) -> MoveOutcome:
        """Compara las piezas detectadas por ArUco contra todas las jugadas legales del turno actual.

        Como el mapa usa símbolos FEN ('P', 'R', 'n', ...), dos piezas del mismo tipo y color
        son 100% intercambiables sin importar qué ID ArUco tengan.
        """
        current_expected = self.get_board_occupancy(self.board)
        if observed_occupancy == current_expected:
            return MoveOutcome(False, "No se detectaron cambios respecto a la posición actual.")

        matched_moves: List[chess.Move] = []
        for legal_move in self.board.legal_moves:
            sim_board = self.board.copy(stack=False)
            sim_board.push(legal_move)
            if self.get_board_occupancy(sim_board) == observed_occupancy:
                matched_moves.append(legal_move)

        if len(matched_moves) == 1:
            return self._commit_move(matched_moves[0])
        if len(matched_moves) > 1:
            # Caso raro (p.ej. promoción sin pieza nueva): prioriza promoción a Dama si aplica
            for mv in matched_moves:
                if mv.promotion == chess.QUEEN:
                    return self._commit_move(mv)
            return self._commit_move(matched_moves[0])

        return MoveOutcome(
            False,
            "La posición observada por cámara no coincide con ninguna jugada legal.",
            audio_events=("invalid_move",),
        )

    def _commit_move(self, move: chess.Move) -> MoveOutcome:
        is_capture = self.board.is_capture(move)
        is_castling = self.board.is_castling(move)
        rook_castling = self._compute_castling_rook_move(move)
        uci = move.uci()

        self.board.push(move)
        events = self._collect_post_move_audio_events()
        return MoveOutcome(
            valid=True,
            message_es=f"Jugada {uci.upper()} aplicada. {self.get_status_text_es()}",
            move_uci=uci,
            is_capture=is_capture,
            is_castling=is_castling,
            rook_castling_move=rook_castling,
            audio_events=events,
        )

    # ------------------------------------------------------------------
    # Motor UCI opcional (Stockfish)
    # ------------------------------------------------------------------
    def start_uci_engine(self, engine_path: Optional[str] = None) -> Tuple[bool, str]:
        path_str = (engine_path if engine_path is not None else self.engine_path).strip()
        if not path_str:
            return False, "No se configuró la ruta del motor UCI (Stockfish)."

        exe = Path(path_str)
        if not exe.exists():
            return False, f"No se encontró el ejecutable del motor en: {exe}"

        self.stop_uci_engine()
        try:
            self._uci_engine = chess.engine.SimpleEngine.popen_uci(str(exe))
            self.engine_path = str(exe)
            return True, f"Motor UCI iniciado: {exe.name}"
        except Exception as exc:
            self._uci_engine = None
            return False, f"No se pudo iniciar el motor UCI: {exc}"

    def compute_best_move(self) -> Tuple[Optional[str], str]:
        if self._uci_engine is None:
            ok, msg = self.start_uci_engine()
            if not ok:
                return None, msg
        assert self._uci_engine is not None
        try:
            result = self._uci_engine.play(
                self.board, chess.engine.Limit(time=max(0.1, float(self.think_time_s)))
            )
            if result.move is None:
                return None, "El motor no devolvió ninguna jugada."
            return result.move.uci(), f"Mejor jugada calculada: {result.move.uci().upper()}"
        except Exception as exc:
            return None, f"Error consultando al motor UCI: {exc}"

    def stop_uci_engine(self) -> None:
        if self._uci_engine is not None:
            try:
                self._uci_engine.quit()
            except Exception:
                pass
            self._uci_engine = None
