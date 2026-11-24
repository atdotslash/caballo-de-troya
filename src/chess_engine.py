"""Motor lógico de ajedrez basado en python-chess.

Funciones principales:
- Mantenimiento y validación estricta del estado legal del tablero (`chess.Board`).
- Detección de jugadas humanas comparando el mapa ArUco observado (piezas intercambiables
  por tipo y color) contra todas las jugadas legales posibles (incluyendo capturas, enroques,
  captura al paso y promociones).
- Integración opcional con un motor UCI local (Stockfish) + oponente IA integrado
  (Minimax con poda Alfa-Beta profundidad 2) como respaldo inmediato sin configuración externa.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine

logger = logging.getLogger(__name__)

PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Bonificación posicional simple por control del centro (casillas d4, e4, d5, e5 y anillo interior)
CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXTENDED_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4, chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}


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
    """Encapsula `python-chess`, un motor UCI opcional (Stockfish) y una IA Minimax integrada."""

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

    def get_legal_destinations(self, from_square_name: str) -> List[str]:
        """Devuelve la lista de casillas destino legales ('e3', 'e4', ...) desde una casilla origen."""
        try:
            src_sq = chess.parse_square(from_square_name.lower())
        except ValueError:
            return []
        destinations = {
            chess.square_name(mv.to_square)
            for mv in self.board.legal_moves
            if mv.from_square == src_sq
        }
        return sorted(destinations)

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

    def apply_move_squares(self, from_sq: str, to_sq: str, promotion: str = "q") -> MoveOutcome:
        """Valida y aplica un movimiento desde casilla origen a destino (promoción automática a Dama)."""
        base_uci = f"{from_sq.strip().lower()}{to_sq.strip().lower()}"
        try:
            candidate = chess.Move.from_uci(base_uci)
            if candidate in self.board.legal_moves:
                return self._commit_move(candidate)
            promo_candidate = chess.Move.from_uci(base_uci + promotion.lower())
            if promo_candidate in self.board.legal_moves:
                return self._commit_move(promo_candidate)
        except ValueError:
            pass
        return MoveOutcome(
            False,
            f"Movimiento inválido o ilegal: {from_sq.upper()} → {to_sq.upper()}.",
            audio_events=("invalid_move",),
        )

    def apply_move_str(self, move_text: str) -> MoveOutcome:
        """Valida y aplica una jugada en notación UCI ('e2e4') o SAN ('Nf3')."""
        clean = move_text.strip()
        if not clean:
            return MoveOutcome(False, "Ingresá una jugada válida (ej. e2e4).", audio_events=("invalid_move",))

        move: Optional[chess.Move] = None
        try:
            candidate = chess.Move.from_uci(clean.lower())
            if candidate in self.board.legal_moves:
                move = candidate
            elif len(clean) == 4:
                promo = chess.Move.from_uci(clean.lower() + "q")
                if promo in self.board.legal_moves:
                    move = promo
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
        """Compara las piezas detectadas por ArUco contra todas las jugadas legales del turno actual."""
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
    # Evaluación heurística y Minimax Alfa-Beta integrado (profundidad 2)
    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate_board(board: chess.Board) -> int:
        """Evaluación estática desde la perspectiva de las Blancas (positivo = mejor para Blancas)."""
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for sq, piece in board.piece_map().items():
            val = PIECE_VALUES[piece.piece_type]
            # Bonificación posicional por control central y avance de peones
            pos_bonus = 0
            if sq in CENTER_SQUARES:
                pos_bonus += 25 if piece.piece_type != chess.KING else -10
            elif sq in EXTENDED_CENTER:
                pos_bonus += 12 if piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.PAWN) else 0
            if piece.piece_type == chess.PAWN:
                rank = chess.square_rank(sq)
                pos_bonus += (rank - 1) * 6 if piece.color == chess.WHITE else (6 - rank) * 6

            total = val + pos_bonus
            score += total if piece.color == chess.WHITE else -total

        return score

    def _minimax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        maximizing_white: bool,
    ) -> int:
        if depth == 0 or board.is_game_over():
            return self._evaluate_board(board)

        moves = list(board.legal_moves)
        # Ordena capturas y jaques primero para acelerar poda alfa-beta
        moves.sort(key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)

        if maximizing_white:
            max_eval = -1000000
            for mv in moves:
                board.push(mv)
                val = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                if val > max_eval:
                    max_eval = val
                if val > alpha:
                    alpha = val
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = 1000000
            for mv in moves:
                board.push(mv)
                val = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                if val < min_eval:
                    min_eval = val
                if val < beta:
                    beta = val
                if beta <= alpha:
                    break
            return min_eval

    def compute_fallback_ai_move(self, depth: int = 2) -> Tuple[Optional[str], str]:
        """Calcula una jugada sólida con Minimax Alfa-Beta sin depender de Stockfish."""
        if self.board.is_game_over():
            return None, "La partida ya finalizó."

        work_board = self.board.copy(stack=False)
        legal_moves = list(work_board.legal_moves)
        if not legal_moves:
            return None, "No hay jugadas legales disponibles."

        random.shuffle(legal_moves)
        maximizing = work_board.turn == chess.WHITE
        best_score = -1000000 if maximizing else 1000000
        best_moves: List[chess.Move] = []

        for mv in legal_moves:
            work_board.push(mv)
            val = self._minimax(work_board, max(1, depth - 1), -1000000, 1000000, not maximizing)
            work_board.pop()
            if maximizing:
                if val > best_score:
                    best_score = val
                    best_moves = [mv]
                elif val == best_score:
                    best_moves.append(mv)
            else:
                if val < best_score:
                    best_score = val
                    best_moves = [mv]
                elif val == best_score:
                    best_moves.append(mv)

        chosen = random.choice(best_moves)
        return chosen.uci(), f"IA interna (Minimax) eligió: {chosen.uci().upper()}"

    # ------------------------------------------------------------------
    # Motor UCI opcional (Stockfish) con respaldo automático Minimax
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

    def compute_best_move(self, allow_fallback: bool = True) -> Tuple[Optional[str], str]:
        if self._uci_engine is None and self.engine_path.strip():
            self.start_uci_engine()

        if self._uci_engine is not None:
            try:
                result = self._uci_engine.play(
                    self.board, chess.engine.Limit(time=max(0.1, float(self.think_time_s)))
                )
                if result.move is not None:
                    return result.move.uci(), f"Stockfish eligió: {result.move.uci().upper()}"
            except Exception as exc:
                logger.warning("Fallo motor UCI, usando IA interna: %s", exc)

        if allow_fallback:
            return self.compute_fallback_ai_move(depth=2)
        return None, "No hay motor UCI configurado."

    def stop_uci_engine(self) -> None:
        if self._uci_engine is not None:
            try:
                self._uci_engine.quit()
            except Exception:
                pass
            self._uci_engine = None
