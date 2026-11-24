"""Módulo de voz offline (TTS es-AR), efectos de sonido (.mp3/.wav) y humor sarcástico.

Características:
- TTS offline vía `pyttsx3` (Windows SAPI5) en un hilo dedicado en segundo plano.
- Reproducción no bloqueante de clips `.mp3` / `.wav` vía `pygame.mixer` sin bloquear UI ni TTS.
- Detección automática de la carpeta existente de sonidos (`assets/sounds/` o `sounds/`)
  sin sobrescribir ni borrar ningún archivo.
- Carga de frases en Español de Argentina desde `assets/frases_humor.json`.
- `play_event(category)`: elige aleatoriamente entre un clip personalizado o una frase TTS
  de la categoría (`inicio`, `demora`, `captura`, `jaque`, `victoria`), evitando repeticiones
  inmediatas.
- Temporizador configurable de demora/inactividad (`delay_warning_seconds`, por defecto 30 s)
  que dispara `play_event("demora")` y se reinicia automáticamente durante el turno humano.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".mp3", ".wav")
HUMOR_CATEGORIES = ("inicio", "demora", "captura", "jaque", "victoria")

EVENT_PHRASES_ES_AR: Dict[str, str] = {
    "startup": "Sistema Caballo de Troya iniciado y listo.",
    "test": "Prueba de voz correcta. Caballo de Troya operativo.",
    "check": "Jaque",
    "checkmate": "Jaque Mate",
    "draw": "Tablas",
    "stalemate": "Tablas por rey ahogado",
    "white_turn": "Turno de las blancas",
    "black_turn": "Turno de las negras",
    "invalid_move": "Movimiento inválido",
    "move_completed": "Jugada completada",
    "clock_pressed": "Reloj presionado",
    "homing_start": "Iniciando búsqueda de origen",
    "homing_done": "Origen establecido correctamente",
    "emergency_stop": "Parada de emergencia activada",
    "calibration_saved": "Calibración del tablero guardada",
    "connected": "Controlador conectado",
    "disconnected": "Controlador desconectado",
    "game_start": "Partida iniciada. Turno de las blancas.",
}


class AudioManager:
    """Administrador híbrido de TTS offline (es-AR) + efectos de sonido + temporizador de demora."""

    def __init__(
        self,
        enabled: bool = True,
        use_tts: bool = False,
        preferred_locale: str = "es-AR",
        rate: int = 175,
        volume: float = 1.0,
        project_root: Optional[str | Path] = None,
        sounds_dir: str | Path = "",
        phrases_path: str | Path = "assets/frases_humor.json",
        humor_enabled: bool = True,
        humor_timer_enabled: bool = True,
        delay_warning_seconds: float = 30.0,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.enabled = enabled
        self.use_tts = bool(use_tts)
        self.preferred_locale = preferred_locale
        self.rate = rate
        self.volume = max(0.0, min(1.0, float(volume)))
        self.humor_enabled = humor_enabled
        self.humor_timer_enabled = humor_timer_enabled
        self.delay_warning_seconds = max(3.0, float(delay_warning_seconds))
        self._on_log = on_log

        self._root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
        self.sounds_dir = self._resolve_sounds_dir(sounds_dir)
        self.phrases_path = self._resolve_path(phrases_path)

        # Inventario dinámico de sonidos existentes y frases JSON
        self._sound_clips: Dict[str, List[Path]] = {cat: [] for cat in HUMOR_CATEGORIES}
        self._humor_phrases: Dict[str, List[str]] = {cat: [] for cat in HUMOR_CATEGORIES}
        self._last_clip: Dict[str, Optional[Path]] = {cat: None for cat in HUMOR_CATEGORIES}
        self._last_phrase: Dict[str, Optional[str]] = {cat: None for cat in HUMOR_CATEGORIES}
        self._last_mode: Dict[str, Optional[str]] = {cat: None for cat in HUMOR_CATEGORIES}

        self.reload_humor_assets()

        # Mixer de pygame para clips .mp3 / .wav sin bloquear UI ni TTS
        self._mixer_lock = threading.RLock()
        self._mixer_ready = False
        self._init_mixer()

        # Temporizador de inactividad/demora durante el turno del jugador humano
        self._timer_lock = threading.RLock()
        self._delay_timer: Optional[threading.Timer] = None
        self._human_turn_active = False

        # Hilo dedicado de TTS offline (solo si use_tts está habilitado)
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._selected_voice_name: str = "Desactivado (Modo Solo Clips MP3)"
        self._available: bool = False
        self._worker: Optional[threading.Thread] = None

        if self.use_tts:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="AudioTTSWorker",
                daemon=True,
            )
            self._worker.start()
        else:
            self._log("[AUDIO] Modo solo clips MP3 activo (TTS pyttsx3 desactivado por configuración).")

    # ------------------------------------------------------------------
    # Descubrimiento y carga de recursos existentes (solo lectura)
    # ------------------------------------------------------------------
    def _resolve_path(self, path_val: str | Path) -> Path:
        p = Path(path_val)
        return p if p.is_absolute() else (self._root / p).resolve()

    def _resolve_sounds_dir(self, configured_dir: str | Path) -> Path:
        """Ubica automáticamente `assets/sounds/` o `sounds/` sin modificar ningún archivo."""
        candidates: List[Path] = []
        if str(configured_dir).strip():
            candidates.append(self._resolve_path(configured_dir))
        candidates.append((self._root / "assets" / "sounds").resolve())
        candidates.append((self._root / "sounds").resolve())

        for cand in candidates:
            if cand.exists() and cand.is_dir():
                return cand
        return (self._root / "assets" / "sounds").resolve()

    def reload_humor_assets(self) -> None:
        """Escanea los .mp3/.wav existentes y carga las frases desde assets/frases_humor.json."""
        total_clips = 0
        for cat in HUMOR_CATEGORIES:
            cat_dir = self.sounds_dir / cat
            clips: List[Path] = []
            if cat_dir.exists() and cat_dir.is_dir():
                for entry in sorted(cat_dir.iterdir()):
                    if entry.is_file() and entry.suffix.lower() in AUDIO_EXTENSIONS:
                        clips.append(entry)
            self._sound_clips[cat] = clips
            total_clips += len(clips)

        total_phrases = 0
        if self.phrases_path.exists():
            try:
                with self.phrases_path.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    for cat in HUMOR_CATEGORIES:
                        items = raw.get(cat, [])
                        if isinstance(items, list):
                            cleaned = [str(x).strip() for x in items if str(x).strip()]
                            self._humor_phrases[cat] = cleaned
                            total_phrases += len(cleaned)
            except Exception as exc:
                self._log(f"[HUMOR] Error cargando {self.phrases_path.name}: {exc}")

        self._log(
            f"[HUMOR] Recursos cargados: {total_clips} clips en '{self.sounds_dir.relative_to(self._root) if self.sounds_dir.is_relative_to(self._root) else self.sounds_dir}' "
            f"y {total_phrases} frases en '{self.phrases_path.name}'."
        )

    def get_sound_clips(self, category: str) -> List[Path]:
        return list(self._sound_clips.get(category.lower().strip(), []))

    def get_humor_phrases(self, category: str) -> List[str]:
        return list(self._humor_phrases.get(category.lower().strip(), []))

    def _init_mixer(self) -> None:
        """Inicializa pygame.mixer sin abrir ventana gráfica."""
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame  # type: ignore

            with self._mixer_lock:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.set_volume(float(self.volume))
                self._mixer_ready = True
            self._log("[AUDIO] Motor pygame.mixer listo para clips .mp3/.wav.")
        except Exception as exc:
            self._mixer_ready = False
            self._log(f"[AUDIO] No se pudo inicializar pygame.mixer: {exc}")

    @property
    def selected_voice_name(self) -> str:
        return self._selected_voice_name

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_mixer_ready(self) -> bool:
        return self._mixer_ready

    def _log(self, msg: str) -> None:
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reproducción de clips (.mp3 / .wav) y eventos de humor
    # ------------------------------------------------------------------
    def play_clip(self, clip_path: Path) -> bool:
        """Reproduce un archivo .mp3 o .wav en segundo plano mediante pygame.mixer.music."""
        if not self.enabled or not self._mixer_ready:
            return False
        if not clip_path.exists():
            self._log(f"[AUDIO] Archivo de sonido no encontrado: {clip_path}")
            return False
        try:
            import pygame  # type: ignore

            with self._mixer_lock:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(str(clip_path))
                pygame.mixer.music.set_volume(float(self.volume))
                pygame.mixer.music.play()
            return True
        except Exception as exc:
            self._log(f"[AUDIO] Error reproduciendo '{clip_path.name}': {exc}")
            return False

    def stop_clip(self) -> None:
        """Detiene el clip .mp3/.wav actual si está sonando."""
        if not self._mixer_ready:
            return
        try:
            import pygame  # type: ignore

            with self._mixer_lock:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
        except Exception:
            pass

    def _pick_non_repeating(self, items: list, last_item: object) -> object:
        if len(items) == 1:
            return items[0]
        candidates = [item for item in items if item != last_item]
        return random.choice(candidates or items)

    def play_event(self, category: str, force_mode: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Reproduce un clip de `assets/sounds/<category>/` (o frase TTS si `use_tts=True`).

        En Modo Solo Audio (`use_tts=False`, predeterminado), reproduce exclusivamente
        los archivos `.mp3`/`.wav` de `assets/sounds/<category>/` evitando repeticiones inmediatas.
        """
        cat = category.lower().strip()
        if cat not in HUMOR_CATEGORIES:
            self._log(f"[HUMOR] Categoría desconocida: '{category}'.")
            return None
        if not self.enabled or not self.humor_enabled:
            return None

        clips = self._sound_clips.get(cat, [])
        phrases = self._humor_phrases.get(cat, [])
        can_clip = bool(clips) and self._mixer_ready
        can_tts = bool(phrases) and self.use_tts

        if not can_clip and not can_tts:
            self._log(f"[HUMOR] Sin clips de audio disponibles para la categoría '{cat}'.")
            return None

        if not self.use_tts:
            mode = "clip"
        elif force_mode in ("clip", "tts"):
            mode = force_mode if ((force_mode == "clip" and can_clip) or (force_mode == "tts" and can_tts)) else ("clip" if can_clip else "tts")
        elif can_clip and can_tts:
            mode = random.choice(("clip", "tts"))
        elif can_clip:
            mode = "clip"
        else:
            mode = "tts"

        self._last_mode[cat] = mode

        if mode == "clip":
            chosen_clip = self._pick_non_repeating(clips, self._last_clip[cat])
            assert isinstance(chosen_clip, Path)
            self._last_clip[cat] = chosen_clip
            if self.play_clip(chosen_clip):
                self._log(f"[HUMOR] Clip ({cat}): {chosen_clip.name}")
                return ("clip", chosen_clip.name)
            if can_tts:
                mode = "tts"
            else:
                return None

        chosen_phrase = self._pick_non_repeating(phrases, self._last_phrase[cat])
        assert isinstance(chosen_phrase, str)
        self._last_phrase[cat] = chosen_phrase
        self._log(f"[HUMOR] Voz ({cat}): {chosen_phrase}")
        self.speak(chosen_phrase, interrupt=False)
        return ("tts", chosen_phrase)

    # ------------------------------------------------------------------
    # Temporizador de inactividad / demora durante el turno humano
    # ------------------------------------------------------------------
    def start_delay_timer(self) -> None:
        """Inicia o reinicia el contador de demora para el turno del jugador humano."""
        with self._timer_lock:
            self._human_turn_active = True
            self._cancel_timer_locked()
            if not self.humor_timer_enabled or not self.enabled or not self.humor_enabled:
                return
            self._schedule_timer_locked()

    def reset_delay_timer(self) -> None:
        """Reinicia el temporizador si el turno humano sigue activo."""
        with self._timer_lock:
            self._cancel_timer_locked()
            if self._human_turn_active and self.humor_timer_enabled and self.enabled and self.humor_enabled:
                self._schedule_timer_locked()

    def stop_delay_timer(self) -> None:
        """Detiene el contador de demora (por ejemplo, cuando mueve el robot o finaliza la partida)."""
        with self._timer_lock:
            self._human_turn_active = False
            self._cancel_timer_locked()

    def set_delay_timer_enabled(self, enabled: bool) -> None:
        with self._timer_lock:
            self.humor_timer_enabled = bool(enabled)
            if not self.humor_timer_enabled:
                self._cancel_timer_locked()
            elif self._human_turn_active and self.enabled and self.humor_enabled:
                self._cancel_timer_locked()
                self._schedule_timer_locked()

    def set_delay_warning_seconds(self, seconds: float) -> None:
        with self._timer_lock:
            self.delay_warning_seconds = max(3.0, float(seconds))
            if self._human_turn_active and self.humor_timer_enabled:
                self._cancel_timer_locked()
                self._schedule_timer_locked()

    def _cancel_timer_locked(self) -> None:
        if self._delay_timer is not None:
            self._delay_timer.cancel()
            self._delay_timer = None

    def _schedule_timer_locked(self) -> None:
        t = threading.Timer(self.delay_warning_seconds, self._on_delay_timeout)
        t.daemon = True
        t.name = "HumorDelayTimer"
        self._delay_timer = t
        t.start()

    def _on_delay_timeout(self) -> None:
        if self._stop_event.is_set():
            return
        should_fire = False
        with self._timer_lock:
            if self._human_turn_active and self.humor_timer_enabled and self.enabled and self.humor_enabled:
                should_fire = True
                self._schedule_timer_locked()
        if should_fire:
            self._log(
                f"[HUMOR] Temporizador de demora ({self.delay_warning_seconds:.0f}s) vencido — disparando evento 'demora'."
            )
            self.play_event("demora")

    # ------------------------------------------------------------------
    # Hilo de síntesis de voz offline (pyttsx3 / Windows SAPI5)
    # ------------------------------------------------------------------
    def _select_spanish_voice(self, engine: object) -> Optional[str]:
        """Prioriza voces es-AR y luego cualquier voz en español instalada en Windows."""
        voices = engine.getProperty("voices") or []
        if not voices:
            return None

        priority_patterns = [
            ("es-ar", "es_ar", "argentina", "elena", "tomas", "tomás"),
            ("es-419", "es-mx", "es-es", "es_", "es-", "spanish", "español", "espanol", "sabina", "helena", "pablo", "laura"),
        ]

        for group in priority_patterns:
            for voice in voices:
                voice_id = str(getattr(voice, "id", "")).lower()
                voice_name = str(getattr(voice, "name", "")).lower()
                langs = " ".join(str(lang).lower() for lang in getattr(voice, "languages", []) or [])
                haystack = f"{voice_id} {voice_name} {langs}"
                if any(token in haystack for token in group):
                    self._selected_voice_name = str(getattr(voice, "name", voice_id))
                    return str(voice.id)

        default_voice = voices[0]
        self._selected_voice_name = str(getattr(default_voice, "name", "Voz predeterminada"))
        return str(default_voice.id)

    def _worker_loop(self) -> None:
        """Inicializa COM/pyttsx3 dentro del hilo de audio y procesa la cola de mensajes."""
        com_initialized = False
        try:
            try:
                import pythoncom  # type: ignore

                pythoncom.CoInitialize()
                com_initialized = True
            except Exception:
                pass

            import pyttsx3  # type: ignore

            engine = pyttsx3.init()
            voice_id = self._select_spanish_voice(engine)
            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.setProperty("rate", int(self.rate))
            engine.setProperty("volume", float(self.volume))
            self._available = True
            self._log(f"[VOZ] Motor TTS offline listo ({self._selected_voice_name}).")
        except Exception as exc:
            self._available = False
            self._selected_voice_name = f"No disponible ({exc})"
            self._log(f"[VOZ] No se pudo inicializar pyttsx3: {exc}")
            return

        try:
            while not self._stop_event.is_set():
                try:
                    text = self._queue.get(timeout=0.3)
                except queue.Empty:
                    continue

                if text is None:
                    break

                if not self.enabled or not text.strip():
                    continue

                try:
                    engine.setProperty("rate", int(self.rate))
                    engine.setProperty("volume", float(self.volume))
                    engine.say(text)
                    engine.runAndWait()
                except Exception as exc:
                    self._log(f"[VOZ] Error al reproducir mensaje: {exc}")
        finally:
            try:
                engine.stop()
            except Exception:
                pass
            if com_initialized:
                try:
                    import pythoncom  # type: ignore

                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def speak(self, text: str, interrupt: bool = False) -> None:
        """Encola una frase para ser hablada en segundo plano (si `use_tts` está habilitado)."""
        if not self.enabled or not self.use_tts or not text or not text.strip():
            return
        if interrupt:
            self.clear_queue()
        self._queue.put(text.strip())

    def announce(self, event_key: str, interrupt: bool = False) -> None:
        """Anuncia un evento predefinido en Español de Argentina (si `use_tts` está habilitado)."""
        if not self.use_tts:
            return
        phrase = EVENT_PHRASES_ES_AR.get(event_key)
        if phrase:
            self.speak(phrase, interrupt=interrupt)

    def clear_queue(self) -> None:
        """Vacía los mensajes pendientes en cola."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        """Detiene limpiamente temporizador, mixer y hilo de voz al cerrar la aplicación."""
        self.stop_delay_timer()
        self.stop_clip()
        self._stop_event.set()
        self.clear_queue()
        self._queue.put(None)
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=1.5)
        try:
            import pygame  # type: ignore

            with self._mixer_lock:
                if pygame.mixer.get_init():
                    pygame.mixer.quit()
        except Exception:
            pass
