"""Pruebas automatizadas de los módulos núcleo de Caballo de Troya (sin hardware físico)."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.audio_manager import HUMOR_CATEGORIES, AudioManager  # noqa: E402
from src.chess_engine import ChessGameEngine  # noqa: E402
from src.config_manager import ConfigManager  # noqa: E402
from src.robot_controller import RobotController  # noqa: E402
from src.vision import ARUCO_PIECE_MAP, BoardCalibrator, VisionSystem  # noqa: E402


class TestConfigManager(unittest.TestCase):
    def test_load_and_atomic_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cm = ConfigManager(cfg_path)
            self.assertTrue(cfg_path.exists())
            self.assertEqual(cm.get("serial", "baudrate"), 115200)
            self.assertEqual(cm.get("audio", "delay_warning_seconds"), 30)
            cm.set("motion", "z_safe_mm", value=52.5)

            cm2 = ConfigManager(cfg_path)
            self.assertEqual(cm2.get("motion", "z_safe_mm"), 52.5)


class TestVisionAndHomography(unittest.TestCase):
    def test_aruco_32_piece_mapping(self) -> None:
        self.assertEqual(len(ARUCO_PIECE_MAP), 32)
        self.assertEqual(ARUCO_PIECE_MAP[0].fen_symbol, "R")
        self.assertEqual(ARUCO_PIECE_MAP[7].fen_symbol, "R")
        self.assertEqual(ARUCO_PIECE_MAP[8].fen_symbol, "P")
        self.assertEqual(ARUCO_PIECE_MAP[15].fen_symbol, "P")
        self.assertEqual(ARUCO_PIECE_MAP[16].fen_symbol, "r")
        self.assertEqual(ARUCO_PIECE_MAP[23].fen_symbol, "r")
        self.assertEqual(ARUCO_PIECE_MAP[20].fen_symbol, "k")
        self.assertEqual(ARUCO_PIECE_MAP[4].fen_symbol, "K")

    def test_homography_and_synthetic_aruco_detection(self) -> None:
        pixel_corners = [[100, 100], [900, 100], [900, 900], [100, 900]]
        machine_corners = [[20.0, 260.0], [260.0, 260.0], [260.0, 20.0], [20.0, 20.0]]
        calibrator = BoardCalibrator(pixel_corners=pixel_corners, machine_corners_mm=machine_corners)
        self.assertTrue(calibrator.is_calibrated)

        ex, ey = calibrator.square_to_machine_mm("e2")
        self.assertAlmostEqual(ex, 155.0, places=2)
        self.assertAlmostEqual(ey, 65.0, places=2)

        img = np.full((1000, 1000, 3), 240, dtype=np.uint8)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        marker = cv2.aruco.generateImageMarker(aruco_dict, 4, 70)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        img[815:885, 515:585] = marker_bgr

        vis = VisionSystem(calibrator=calibrator)
        detections = vis.detect_pieces(img)
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.marker_id, 4)
        self.assertIsNotNone(det.piece)
        assert det.piece is not None
        self.assertEqual(det.piece.name_es, "Rey Blanco")
        self.assertEqual(det.square, "e1")
        self.assertIsNotNone(det.machine_mm)
        assert det.machine_mm is not None
        self.assertAlmostEqual(det.machine_mm[0], 155.0, delta=1.5)
        self.assertAlmostEqual(det.machine_mm[1], 35.0, delta=1.5)


class TestRobotControllerGCodeAndSimulation(unittest.TestCase):
    def test_simultaneous_gcode_and_clock_sequence(self) -> None:
        cm = ConfigManager(ROOT_DIR / "config.json")
        emitted: list[str] = []

        ctrl = RobotController(config_getter=lambda: cm.data)
        ctrl.enqueue_gcode = lambda gcode, timeout_s=None, wait=False: (emitted.append(gcode.strip()) or True, "ok")  # type: ignore

        ctrl.move_simultaneous(50.0, 30.0, 10.0, rapid=True)
        self.assertEqual(emitted[-1], "G0 X50.000 Y30.000 Z10.000")

        emitted.clear()
        ctrl.press_chess_clock(return_standby=True)
        self.assertTrue(any(line.startswith("G0 X280.000 Y140.000 Z45.000") for line in emitted))
        self.assertTrue(any(line.startswith("G1 X280.000 Y140.000 Z12.000") for line in emitted))
        self.assertTrue(any(line.startswith("G4 P") for line in emitted))
        self.assertEqual(emitted[-1], "G0 X10.000 Y10.000 Z45.000")

    def test_simulation_mode_realtime_mpos_and_sim_logs(self) -> None:
        cm = ConfigManager(ROOT_DIR / "config.json")
        logs: list[tuple[str, str]] = []
        ctrl = RobotController(
            config_getter=lambda: cm.data,
            on_log=lambda d, m: logs.append((d, m)),
        )
        try:
            ok, _ = ctrl.connect_simulation()
            self.assertTrue(ok)
            self.assertTrue(ctrl.is_connected)
            self.assertTrue(ctrl.is_simulation)

            # Movimiento simultáneo G0
            ok_move, resp = ctrl.enqueue_gcode("G0 X50.000 Y30.000 Z15.000", wait=True)
            self.assertTrue(ok_move)
            self.assertEqual(resp, "ok")
            self.assertAlmostEqual(ctrl.status.x, 50.0, places=2)
            self.assertAlmostEqual(ctrl.status.y, 30.0, places=2)
            self.assertAlmostEqual(ctrl.status.z, 15.0, places=2)
            self.assertTrue(ctrl.status.simulated)

            # Jog incremental $J=G91
            ok_jog, resp_jog = ctrl.enqueue_gcode("$J=G91 G21 X10.0 Y-5.0 Z2.5 F1800", wait=True)
            self.assertTrue(ok_jog)
            self.assertEqual(resp_jog, "ok")
            self.assertAlmostEqual(ctrl.status.x, 60.0, places=2)
            self.assertAlmostEqual(ctrl.status.y, 25.0, places=2)
            self.assertAlmostEqual(ctrl.status.z, 17.5, places=2)

            # Servo garra M3/M5
            ok_m3, _ = ctrl.enqueue_gcode("M3 S150", wait=True)
            ok_m5, _ = ctrl.enqueue_gcode("M5", wait=True)
            self.assertTrue(ok_m3 and ok_m5)

            # Homing $H vuelve a (0,0,0)
            ok_h, _ = ctrl.enqueue_gcode("$H", wait=True)
            self.assertTrue(ok_h)
            self.assertAlmostEqual(ctrl.status.x, 0.0, places=2)
            self.assertAlmostEqual(ctrl.status.y, 0.0, places=2)
            self.assertAlmostEqual(ctrl.status.z, 0.0, places=2)

            # Verifica que se registró con prefijo SIM
            sim_logs = [m for d, m in logs if d == "SIM"]
            self.assertTrue(any(">> G0 X50.000 Y30.000 Z15.000" in m for m in sim_logs))
            self.assertTrue(any("<< ok" in m for m in sim_logs))
        finally:
            ctrl.disconnect()


class TestChessEngine(unittest.TestCase):
    def test_vision_move_detection_and_checkmate(self) -> None:
        eng = ChessGameEngine()
        occ = eng.get_board_occupancy()
        occ.pop("e2")
        occ["e4"] = "P"
        outcome = eng.detect_human_move_from_vision(occ)
        self.assertTrue(outcome.valid)
        self.assertEqual(outcome.move_uci, "e2e4")
        self.assertIn("black_turn", outcome.audio_events)

        eng.reset_game()
        self.assertTrue(eng.apply_move_str("f2f3").valid)
        self.assertTrue(eng.apply_move_str("e7e5").valid)
        self.assertTrue(eng.apply_move_str("g2g4").valid)
        mate = eng.apply_move_str("d8h4")
        self.assertTrue(mate.valid)
        self.assertIn("checkmate", mate.audio_events)


class TestAudioAndHumor(unittest.TestCase):
    def test_humor_json_counts_and_existing_mp3_discovery_and_playback(self) -> None:
        phrases_file = ROOT_DIR / "assets" / "frases_humor.json"
        self.assertTrue(phrases_file.exists())
        data = json.loads(phrases_file.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data["inicio"]), 15)
        self.assertGreaterEqual(len(data["victoria"]), 15)
        self.assertGreaterEqual(len(data["demora"]), 25)
        self.assertGreaterEqual(len(data["captura"]), 25)
        self.assertGreaterEqual(len(data["jaque"]), 25)

        audio = AudioManager(
            enabled=True,
            volume=0.0,  # volumen 0 para pruebas automatizadas silenciosas
            project_root=ROOT_DIR,
            humor_enabled=True,
            humor_timer_enabled=False,
        )
        try:
            self.assertTrue(audio.is_mixer_ready, "pygame.mixer debería estar inicializado")

            # Verifica descubrimiento de .mp3 existentes en las 5 carpetas sin borrar nada
            for cat in HUMOR_CATEGORIES:
                clips = audio.get_sound_clips(cat)
                self.assertGreater(len(clips), 0, f"Debería haber clips .mp3 en {cat}")

            # Prueba carga y reproducción real de un .mp3 por categoría con pygame.mixer
            for cat in HUMOR_CATEGORIES:
                res = audio.play_event(cat, force_mode="clip")
                self.assertIsNotNone(res)
                assert res is not None
                self.assertEqual(res[0], "clip")
                self.assertTrue(res[1].lower().endswith((".mp3", ".wav")))
            audio.stop_clip()

            # Verifica que evita repeticiones inmediatas en frases TTS
            first = audio.play_event("jaque", force_mode="tts")
            second = audio.play_event("jaque", force_mode="tts")
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertNotEqual(first, second)
        finally:
            audio.shutdown()

    def test_inactivity_delay_timer_triggers_demora(self) -> None:
        fired = threading.Event()
        audio = AudioManager(
            enabled=True,
            volume=0.0,
            project_root=ROOT_DIR,
            humor_enabled=True,
            humor_timer_enabled=True,
            delay_warning_seconds=3.0,
        )
        try:
            # Reduce internamente el intervalo solo para la prueba unitaria rápida
            audio.delay_warning_seconds = 0.25
            original_play = audio.play_event

            def spy_play(cat: str, force_mode: str | None = None):
                if cat == "demora":
                    fired.set()
                return original_play(cat, force_mode="clip")

            audio.play_event = spy_play  # type: ignore
            audio.start_delay_timer()
            self.assertTrue(fired.wait(timeout=1.5), "El temporizador debería disparar 'demora'")
            audio.stop_delay_timer()
        finally:
            audio.shutdown()


if __name__ == "__main__":
    unittest.main()
