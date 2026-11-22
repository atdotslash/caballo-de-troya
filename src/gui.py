"""Interfaz gráfica de escritorio en CustomTkinter (100% Español de Argentina).

Paneles incluidos:
1. Barra superior de conexión serial (detección de puertos COM, Conectar/Desconectar,
   Homing $H, Desbloquear $X, Parada de Emergencia !, Reiniciar Ctrl-X y telemetría X/Y/Z).
2. Visor de cámara cenital en vivo con cajas ArUco, nombres de piezas en español,
   grilla 8x8 calibrada por homografía y asistente interactivo de calibración de 4 esquinas.
3. Panel lateral con pestañas:
   - Calibración y Pruebas (Jog X/Y/Z, Movimiento Simultáneo G0 X Y Z, Garra SG90 D11,
     Reloj de Ajedrez, Perfiles GRBL $110-$122, Prueba de Voz es-AR).
   - Tablero y Partida (estado python-chess, detección por visión, ejecución física manual).
   - Configuración (alturas Z, coordenadas del reloj, posición de espera y esquinas físicas en mm).
4. Consola G-code en vivo con envío manual de comandos.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.audio_manager import EVENT_PHRASES_ES_AR, AudioManager
from src.chess_engine import ChessGameEngine
from src.config_manager import ConfigManager
from src.robot_controller import MachineStatus, RobotController
from src.vision import CORNER_INSTRUCTIONS_ES, BoardCalibrator, DetectedPiece, VisionSystem

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


VOICE_TEST_LABELS_ES: List[Tuple[str, str]] = [
    ("Prueba de sistema", "test"),
    ("Jaque", "check"),
    ("Jaque Mate", "checkmate"),
    ("Tablas", "draw"),
    ("Turno de las blancas", "white_turn"),
    ("Turno de las negras", "black_turn"),
    ("Movimiento inválido", "invalid_move"),
    ("Reloj presionado", "clock_pressed"),
    ("Parada de emergencia", "emergency_stop"),
]


class ChessRobotApp(ctk.CTk):
    """Ventana principal de Caballo de Troya."""

    def __init__(self, config_path: str = "config.json") -> None:
        super().__init__()

        self.title("Caballo de Troya — Brazo Robótico de Ajedrez SCARA")
        self.geometry("1440x880")
        self.minsize(1240, 760)

        # Colas thread-safe para actualizar la UI desde hilos de serial/audio
        self._log_queue: queue.Queue[Tuple[str, str]] = queue.Queue()
        self._status_queue: queue.Queue[MachineStatus] = queue.Queue()
        self._conn_queue: queue.Queue[Tuple[bool, str]] = queue.Queue()

        # Módulos núcleo
        self.config_manager = ConfigManager(config_path)
        self._init_subsystems()

        # Estado UI
        self._com_display_to_device: Dict[str, str] = {}
        self._jog_step_mm = ctk.DoubleVar(value=10.0)
        self._show_grid_var = ctk.BooleanVar(value=True)
        self._show_aruco_var = ctk.BooleanVar(value=True)
        self._gripper_pwm_var = ctk.IntVar(
            value=int(self.config_manager.get("gripper", "open_pwm", default=65))
        )
        self._camera_idx_var = ctk.StringVar(
            value=str(self.config_manager.get("vision", "camera_index", default=0))
        )
        self._sim_mode_var = ctk.BooleanVar(value=False)
        self._humor_timer_var = ctk.BooleanVar(
            value=bool(self.config_manager.get("audio", "humor_timer_enabled", default=True))
        )

        # Estado de calibración interactiva de 4 esquinas
        self._calibrating = False
        self._pending_calib_pts: List[Tuple[float, float]] = []
        self._last_frame_shape: Optional[Tuple[int, int]] = None  # (width, height)
        self._display_transform: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.0)
        self._video_photo: Optional[ImageTk.PhotoImage] = None
        self._last_detections: List[DetectedPiece] = []

        # Construcción de la interfaz
        self._build_ui()
        self._refresh_com_ports()
        self._sync_config_fields_to_ui()
        self._update_chess_panel()

        # Arranque de cámara si es posible (sin bloquear)
        self._log_to_console("SYS", "Aplicación iniciada. Revisá el puerto COM e iniciá la cámara para comenzar.")
        self.after(300, self._try_autostart_camera)
        self.after(40, self._ui_tick)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Inicialización de subsistemas
    # ------------------------------------------------------------------
    def _init_subsystems(self) -> None:
        audio_cfg = self.config_manager.get("audio", default={})
        self.audio = AudioManager(
            enabled=bool(audio_cfg.get("enabled", True)),
            preferred_locale=str(audio_cfg.get("preferred_locale", "es-AR")),
            rate=int(audio_cfg.get("rate", 175)),
            volume=float(audio_cfg.get("volume", 1.0)),
            sounds_dir=str(audio_cfg.get("sounds_dir", "")),
            phrases_path=str(audio_cfg.get("phrases_path", "assets/frases_humor.json")),
            humor_enabled=bool(audio_cfg.get("humor_enabled", True)),
            humor_timer_enabled=bool(audio_cfg.get("humor_timer_enabled", True)),
            delay_warning_seconds=float(audio_cfg.get("delay_warning_seconds", 30.0)),
            on_log=lambda msg: self._log_queue.put(("SYS", msg)),
        )

        vis_cfg = self.config_manager.get("vision", default={})
        calib_cfg = vis_cfg.get("calibration", {})
        calibrator = BoardCalibrator(
            pixel_corners=calib_cfg.get("pixel_corners") if calib_cfg.get("calibrated") else None,
            machine_corners_mm=calib_cfg.get("machine_corners_mm"),
        )
        self.vision = VisionSystem(
            camera_index=int(vis_cfg.get("camera_index", 0)),
            frame_width=int(vis_cfg.get("frame_width", 1280)),
            frame_height=int(vis_cfg.get("frame_height", 720)),
            calibrator=calibrator,
        )

        self.robot = RobotController(
            config_getter=lambda: self.config_manager.data,
            on_log=lambda direction, msg: self._log_queue.put((direction, msg)),
            on_status=lambda status: self._status_queue.put(status),
            on_connection_change=lambda conn, port: self._conn_queue.put((conn, port)),
        )

        chess_cfg = self.config_manager.get("chess", default={})
        self.chess_engine = ChessGameEngine(
            engine_path=str(chess_cfg.get("engine_path", "")),
            think_time_s=float(chess_cfg.get("engine_think_time_s", 1.0)),
        )

    # ------------------------------------------------------------------
    # Construcción de Layout
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_top_serial_bar()

        # Cuerpo principal: izquierda (cámara) + derecha (pestañas)
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 6))
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=2, minsize=460)
        main_frame.grid_rowconfigure(0, weight=1)

        self._build_vision_panel(main_frame)
        self._build_right_tabs(main_frame)
        self._build_console_panel()

    def _build_top_serial_bar(self) -> None:
        bar = ctk.CTkFrame(self, corner_radius=8)
        bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            bar, text="Puerto COM:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4), pady=8)

        self.com_combo = ctk.CTkComboBox(bar, width=230, state="readonly", values=["Sin puertos"])
        self.com_combo.pack(side="left", padx=4, pady=8)

        ctk.CTkButton(
            bar, text="Actualizar", width=85, command=self._refresh_com_ports
        ).pack(side="left", padx=4, pady=8)

        self.btn_connect = ctk.CTkButton(
            bar,
            text="Conectar",
            width=105,
            fg_color="#1f6aa5",
            command=self._toggle_connection,
        )
        self.btn_connect.pack(side="left", padx=4, pady=8)

        self.sw_sim_mode = ctk.CTkSwitch(
            bar,
            text="Modo Simulación",
            variable=self._sim_mode_var,
            command=self._on_toggle_simulation_mode,
            font=ctk.CTkFont(size=12, weight="bold"),
            progress_color="#7c3aed",
        )
        self.sw_sim_mode.pack(side="left", padx=(10, 4), pady=8)

        ctk.CTkButton(
            bar,
            text="Hacer Homing ($H)",
            width=140,
            fg_color="#2d7d46",
            hover_color="#236337",
            command=self._on_homing,
        ).pack(side="left", padx=(12, 4), pady=8)

        ctk.CTkButton(
            bar,
            text="Desbloquear ($X)",
            width=125,
            fg_color="#4a5568",
            hover_color="#394250",
            command=self._on_unlock,
        ).pack(side="left", padx=4, pady=8)

        ctk.CTkButton(
            bar,
            text="Parada de Emergencia (!)",
            width=185,
            fg_color="#c62828",
            hover_color="#9a1d1d",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_emergency_stop,
        ).pack(side="left", padx=(12, 4), pady=8)

        ctk.CTkButton(
            bar,
            text="Reiniciar",
            width=90,
            fg_color="#b26a00",
            hover_color="#8a5200",
            command=self._on_soft_reset,
        ).pack(side="left", padx=4, pady=8)

        # Telemetría GRBL en tiempo real (lado derecho)
        telemetry_box = ctk.CTkFrame(bar, fg_color="#1a1d24", corner_radius=6)
        telemetry_box.pack(side="right", padx=10, pady=6)

        self.lbl_grbl_state = ctk.CTkLabel(
            telemetry_box,
            text="Estado: Desconectado",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f59e0b",
        )
        self.lbl_grbl_state.pack(side="left", padx=(10, 12), pady=4)

        self.lbl_grbl_pos = ctk.CTkLabel(
            telemetry_box,
            text="X: 0.00  Y: 0.00  Z: 0.00 mm",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.lbl_grbl_pos.pack(side="left", padx=(0, 10), pady=4)

    def _build_vision_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, corner_radius=8)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(2, weight=1)

        # Barra superior de controles de cámara
        cam_bar = ctk.CTkFrame(panel, fg_color="transparent")
        cam_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            cam_bar, text="Cámara Cenital:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(0, 6))

        ctk.CTkComboBox(
            cam_bar,
            width=64,
            state="readonly",
            values=["0", "1", "2", "3"],
            variable=self._camera_idx_var,
        ).pack(side="left", padx=4)

        self.btn_camera = ctk.CTkButton(
            cam_bar, text="Iniciar Cámara", width=120, command=self._toggle_camera
        )
        self.btn_camera.pack(side="left", padx=4)

        self.btn_calibrate = ctk.CTkButton(
            cam_bar,
            text="Calibrar Tablero",
            width=135,
            fg_color="#0e7490",
            hover_color="#0c5f75",
            command=self._toggle_calibration_mode,
        )
        self.btn_calibrate.pack(side="left", padx=(10, 4))

        ctk.CTkButton(
            cam_bar,
            text="Limpiar Calibración",
            width=130,
            fg_color="#475569",
            hover_color="#334155",
            command=self._clear_calibration,
        ).pack(side="left", padx=4)

        ctk.CTkCheckBox(
            cam_bar, text="Grilla 8x8", variable=self._show_grid_var, width=85
        ).pack(side="right", padx=6)
        ctk.CTkCheckBox(
            cam_bar, text="ArUco", variable=self._show_aruco_var, width=70
        ).pack(side="right", padx=4)

        # Banner de guía de calibración (se resalta cuando está calibrando)
        self.lbl_calib_banner = ctk.CTkLabel(
            panel,
            text="Vista en vivo — Presioná 'Calibrar Tablero' para marcar las 4 esquinas del área 8x8.",
            fg_color="#1f2937",
            corner_radius=6,
            height=28,
            font=ctk.CTkFont(size=12),
        )
        self.lbl_calib_banner.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 6))

        # Contenedor de video interactivo
        self.video_container = ctk.CTkFrame(panel, fg_color="#0f1117", corner_radius=6)
        self.video_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 6))
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)

        self.video_label = tk.Label(
            self.video_container,
            bg="#0f1117",
            fg="#94a3b8",
            text="Cámara inactiva.\nSeleccioná un índice y hacé clic en 'Iniciar Cámara'.",
            font=("Segoe UI", 12),
            cursor="crosshair",
            bd=0,
            highlightthickness=0,
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<Button-1>", self._on_video_click)

        # Barra inferior de métricas de visión
        info_bar = ctk.CTkFrame(panel, fg_color="transparent")
        info_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))

        self.lbl_vision_stats = ctk.CTkLabel(
            info_bar,
            text="Piezas detectadas: 0 / 32  |  FPS: 0.0  |  Calibración: Pendiente",
            font=ctk.CTkFont(size=12),
            text_color="#cbd5e1",
        )
        self.lbl_vision_stats.pack(side="left")

    def _build_right_tabs(self, parent: ctk.CTkFrame) -> None:
        self.tabs = ctk.CTkTabview(parent, corner_radius=8)
        self.tabs.grid(row=0, column=1, sticky="nsew")

        tab_calib = self.tabs.add("Calibración y Pruebas")
        tab_game = self.tabs.add("Tablero y Partida")
        tab_cfg = self.tabs.add("Configuración")

        self._build_calibration_tab(tab_calib)
        self._build_game_tab(tab_game)
        self._build_config_tab(tab_cfg)

    def _build_calibration_tab(self, parent: ctk.CTkFrame) -> None:
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # ---- 1. Control Manual (Jog X/Y/Z) ----
        jog_card = ctk.CTkFrame(scroll, corner_radius=8)
        jog_card.pack(fill="x", padx=4, pady=(2, 8))

        ctk.CTkLabel(
            jog_card,
            text="Control Manual (Jog X / Y / Z)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        step_row = ctk.CTkFrame(jog_card, fg_color="transparent")
        step_row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(step_row, text="Paso (mm):").pack(side="left", padx=(0, 6))
        self.step_selector = ctk.CTkSegmentedButton(
            step_row,
            values=["0.5", "1", "5", "10", "25", "50"],
            command=lambda v: self._jog_step_mm.set(float(v)),
        )
        self.step_selector.set("10")
        self.step_selector.pack(side="left", fill="x", expand=True)

        pad_frame = ctk.CTkFrame(jog_card, fg_color="transparent")
        pad_frame.pack(fill="x", padx=10, pady=6)

        # Cruceta X/Y
        xy_grid = ctk.CTkFrame(pad_frame, fg_color="transparent")
        xy_grid.pack(side="left", padx=(0, 14))
        ctk.CTkButton(xy_grid, text="Y+", width=68, command=lambda: self._jog(0, 1, 0)).grid(
            row=0, column=1, padx=3, pady=3
        )
        ctk.CTkButton(xy_grid, text="X-", width=68, command=lambda: self._jog(-1, 0, 0)).grid(
            row=1, column=0, padx=3, pady=3
        )
        ctk.CTkButton(
            xy_grid,
            text="Espera",
            width=68,
            fg_color="#475569",
            command=self._on_go_standby,
        ).grid(row=1, column=1, padx=3, pady=3)
        ctk.CTkButton(xy_grid, text="X+", width=68, command=lambda: self._jog(1, 0, 0)).grid(
            row=1, column=2, padx=3, pady=3
        )
        ctk.CTkButton(xy_grid, text="Y-", width=68, command=lambda: self._jog(0, -1, 0)).grid(
            row=2, column=1, padx=3, pady=3
        )

        # Columna Z
        z_col = ctk.CTkFrame(pad_frame, fg_color="transparent")
        z_col.pack(side="left", padx=6)
        ctk.CTkButton(
            z_col, text="Z+ (Subir)", width=110, fg_color="#0f766e", command=lambda: self._jog(0, 0, 1)
        ).pack(pady=3)
        ctk.CTkButton(
            z_col, text="Ir a Z Seguro", width=110, fg_color="#475569", command=self._on_go_safe_z
        ).pack(pady=3)
        ctk.CTkButton(
            z_col, text="Z- (Bajar)", width=110, fg_color="#0f766e", command=lambda: self._jog(0, 0, -1)
        ).pack(pady=3)

        # ---- 2. Prueba de Movimiento Simultáneo (G0 X Y Z) ----
        sim_card = ctk.CTkFrame(scroll, corner_radius=8)
        sim_card.pack(fill="x", padx=4, pady=(0, 8))

        ctk.CTkLabel(
            sim_card,
            text="Movimiento Simultáneo 3 Ejes (G0 X Y Z)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        sim_row = ctk.CTkFrame(sim_card, fg_color="transparent")
        sim_row.pack(fill="x", padx=10, pady=(2, 8))

        ctk.CTkLabel(sim_row, text="X:").pack(side="left")
        self.ent_sim_x = ctk.CTkEntry(sim_row, width=62)
        self.ent_sim_x.insert(0, "100.0")
        self.ent_sim_x.pack(side="left", padx=(3, 8))

        ctk.CTkLabel(sim_row, text="Y:").pack(side="left")
        self.ent_sim_y = ctk.CTkEntry(sim_row, width=62)
        self.ent_sim_y.insert(0, "100.0")
        self.ent_sim_y.pack(side="left", padx=(3, 8))

        ctk.CTkLabel(sim_row, text="Z:").pack(side="left")
        self.ent_sim_z = ctk.CTkEntry(sim_row, width=62)
        self.ent_sim_z.insert(0, str(self.config_manager.get("motion", "z_safe_mm", default=45.0)))
        self.ent_sim_z.pack(side="left", padx=(3, 8))

        ctk.CTkButton(
            sim_row, text="Mover Simultáneo", width=130, command=self._on_move_simultaneous
        ).pack(side="right")

        # ---- 3. Control de Garra SG90 (Pin D11) ----
        grip_card = ctk.CTkFrame(scroll, corner_radius=8)
        grip_card.pack(fill="x", padx=4, pady=(0, 8))

        grip_header = ctk.CTkFrame(grip_card, fg_color="transparent")
        grip_header.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(
            grip_header,
            text="Garra Servo SG90 (Pin D11 — M3 S<valor>)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")
        self.lbl_gripper_val = ctk.CTkLabel(
            grip_header,
            text=f"S{self._gripper_pwm_var.get()}",
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            text_color="#38bdf8",
        )
        self.lbl_gripper_val.pack(side="right")

        min_pwm = int(self.config_manager.get("gripper", "min_pwm", default=0))
        max_pwm = int(self.config_manager.get("gripper", "max_pwm", default=255))
        self.slider_gripper = ctk.CTkSlider(
            grip_card,
            from_=min_pwm,
            to=max_pwm,
            number_of_steps=max(1, max_pwm - min_pwm),
            command=self._on_gripper_slider_drag,
        )
        self.slider_gripper.set(self._gripper_pwm_var.get())
        self.slider_gripper.pack(fill="x", padx=12, pady=4)

        grip_btns_1 = ctk.CTkFrame(grip_card, fg_color="transparent")
        grip_btns_1.pack(fill="x", padx=10, pady=3)
        ctk.CTkButton(
            grip_btns_1, text="Aplicar Slider", width=100, command=self._on_send_gripper_slider
        ).pack(side="left", padx=2, expand=True, fill="x")
        ctk.CTkButton(
            grip_btns_1, text="Abrir Garra", width=95, command=self._on_open_gripper
        ).pack(side="left", padx=2, expand=True, fill="x")
        ctk.CTkButton(
            grip_btns_1, text="Cerrar Garra", width=95, command=self._on_close_gripper
        ).pack(side="left", padx=2, expand=True, fill="x")

        grip_btns_2 = ctk.CTkFrame(grip_card, fg_color="transparent")
        grip_btns_2.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkButton(
            grip_btns_2,
            text="Probar Garra",
            fg_color="#0e7490",
            command=self._on_test_gripper,
        ).pack(side="left", padx=2, expand=True, fill="x")
        ctk.CTkButton(
            grip_btns_2,
            text="Apagar PWM (M5)",
            fg_color="#475569",
            command=lambda: self.robot.disable_gripper_pwm(),
        ).pack(side="left", padx=2, expand=True, fill="x")
        ctk.CTkButton(
            grip_btns_2,
            text="Guardar Abierta",
            fg_color="#334155",
            command=lambda: self._save_gripper_preset("open_pwm"),
        ).pack(side="left", padx=2, expand=True, fill="x")
        ctk.CTkButton(
            grip_btns_2,
            text="Guardar Cerrada",
            fg_color="#334155",
            command=lambda: self._save_gripper_preset("close_pwm"),
        ).pack(side="left", padx=2, expand=True, fill="x")

        # ---- 4. Reloj de Ajedrez ----
        clock_card = ctk.CTkFrame(scroll, corner_radius=8)
        clock_card.pack(fill="x", padx=4, pady=(0, 8))

        ctk.CTkLabel(
            clock_card,
            text="Reloj de Ajedrez",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 2))

        self.lbl_clock_coords = ctk.CTkLabel(
            clock_card,
            text="Coordenada guardada: X=0.0  Y=0.0  Z=0.0 mm",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color="#cbd5e1",
        )
        self.lbl_clock_coords.pack(anchor="w", padx=10, pady=(0, 6))

        clock_btns = ctk.CTkFrame(clock_card, fg_color="transparent")
        clock_btns.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkButton(
            clock_btns,
            text="Guardar Pos. Actual como Reloj",
            fg_color="#475569",
            command=self._on_save_clock_from_current,
        ).pack(side="left", padx=(0, 4), expand=True, fill="x")
        ctk.CTkButton(
            clock_btns,
            text="Presionar Reloj",
            fg_color="#2d7d46",
            hover_color="#236337",
            command=self._on_press_clock,
        ).pack(side="left", padx=(4, 0), expand=True, fill="x")

        # ---- 5. Perfiles GRBL ($110-$122) ----
        preset_card = ctk.CTkFrame(scroll, corner_radius=8)
        preset_card.pack(fill="x", padx=4, pady=(0, 8))

        ctk.CTkLabel(
            preset_card,
            text="Perfiles de Velocidad y Aceleración GRBL ($110–$122)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        preset_row = ctk.CTkFrame(preset_card, fg_color="transparent")
        preset_row.pack(fill="x", padx=10, pady=(0, 8))

        presets = self.config_manager.get("grbl_presets", "presets", default={})
        self._preset_label_to_key = {
            str(v.get("label", k)): k for k, v in presets.items()
        }
        active_key = self.config_manager.get("grbl_presets", "active_preset", default="normal")
        active_label = str(presets.get(active_key, {}).get("label", "Normal (Partida)"))

        self.preset_combo = ctk.CTkComboBox(
            preset_row,
            state="readonly",
            values=list(self._preset_label_to_key.keys()) or ["Normal (Partida)"],
        )
        self.preset_combo.set(active_label)
        self.preset_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            preset_row,
            text="Aplicar Perfil GRBL",
            width=140,
            command=self._on_apply_grbl_preset,
        ).pack(side="right")

        # ---- 6. Prueba de Voz (es-AR) ----
        voice_card = ctk.CTkFrame(scroll, corner_radius=8)
        voice_card.pack(fill="x", padx=4, pady=(0, 8))

        ctk.CTkLabel(
            voice_card,
            text="Síntesis de Voz Offline (Español Argentina)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        voice_row = ctk.CTkFrame(voice_card, fg_color="transparent")
        voice_row.pack(fill="x", padx=10, pady=(0, 8))

        self.voice_combo = ctk.CTkComboBox(
            voice_row,
            state="readonly",
            values=[label for label, _ in VOICE_TEST_LABELS_ES],
        )
        self.voice_combo.set(VOICE_TEST_LABELS_ES[0][0])
        self.voice_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            voice_row,
            text="Probar Voz",
            width=110,
            command=self._on_test_voice,
        ).pack(side="right")

        # ---- 7. Probar Humor y Sonidos ----
        humor_card = ctk.CTkFrame(scroll, corner_radius=8)
        humor_card.pack(fill="x", padx=4, pady=(0, 8))

        humor_hdr = ctk.CTkFrame(humor_card, fg_color="transparent")
        humor_hdr.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(
            humor_hdr,
            text="Probar Humor y Sonidos",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")

        ctk.CTkButton(
            humor_hdr,
            text="Detener Audio",
            width=100,
            height=24,
            fg_color="#475569",
            command=self.audio.stop_clip,
        ).pack(side="right")

        humor_btns_1 = ctk.CTkFrame(humor_card, fg_color="transparent")
        humor_btns_1.pack(fill="x", padx=10, pady=(2, 3))
        for label, cat, color in (
            ("Inicio", "inicio", "#0e7490"),
            ("Demora", "demora", "#b45309"),
            ("Captura", "captura", "#7c3aed"),
        ):
            ctk.CTkButton(
                humor_btns_1,
                text=label,
                fg_color=color,
                command=lambda c=cat: self._on_test_humor_category(c),
            ).pack(side="left", padx=2, expand=True, fill="x")

        humor_btns_2 = ctk.CTkFrame(humor_card, fg_color="transparent")
        humor_btns_2.pack(fill="x", padx=10, pady=(2, 6))
        for label, cat, color in (
            ("Jaque", "jaque", "#b91c1c"),
            ("Victoria", "victoria", "#2d7d46"),
        ):
            ctk.CTkButton(
                humor_btns_2,
                text=label,
                fg_color=color,
                command=lambda c=cat: self._on_test_humor_category(c),
            ).pack(side="left", padx=2, expand=True, fill="x")

        delay_s = int(round(float(self.config_manager.get("audio", "delay_warning_seconds", default=30))))
        self.sw_humor_timer = ctk.CTkSwitch(
            humor_card,
            text=f"Activar temporizador de demora ({delay_s}s)",
            variable=self._humor_timer_var,
            command=self._on_toggle_humor_timer,
        )
        self.sw_humor_timer.pack(anchor="w", padx=12, pady=(2, 8))

    def _build_game_tab(self, parent: ctk.CTkFrame) -> None:
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=4, pady=4)

        self.lbl_game_status = ctk.CTkLabel(
            container,
            text="Turno de las Blancas",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f2937",
            corner_radius=6,
            height=34,
        )
        self.lbl_game_status.pack(fill="x", pady=(0, 8))

        self.txt_board_ascii = ctk.CTkTextbox(
            container, height=205, font=ctk.CTkFont(family="Consolas", size=14)
        )
        self.txt_board_ascii.pack(fill="x", pady=(0, 8))
        self.txt_board_ascii.configure(state="disabled")

        move_frame = ctk.CTkFrame(container, corner_radius=8)
        move_frame.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            move_frame,
            text="Ejecutar Jugada Manual (UCI ej: e2e4)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        row = ctk.CTkFrame(move_frame, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 8))
        self.ent_manual_move = ctk.CTkEntry(row, placeholder_text="Ej: e2e4")
        self.ent_manual_move.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.ent_manual_move.bind("<Return>", lambda _e: self._on_execute_manual_move())

        ctk.CTkButton(
            row,
            text="Ejecutar en Robot",
            width=145,
            fg_color="#2d7d46",
            hover_color="#236337",
            command=self._on_execute_manual_move,
        ).pack(side="right")

        vis_btns = ctk.CTkFrame(container, fg_color="transparent")
        vis_btns.pack(fill="x", pady=4)
        ctk.CTkButton(
            vis_btns,
            text="Detectar Jugada por Cámara",
            command=self._on_detect_move_from_vision,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(
            vis_btns,
            text="Reiniciar Partida",
            fg_color="#475569",
            command=self._on_reset_game,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _build_config_tab(self, parent: ctk.CTkFrame) -> None:
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._cfg_entries: Dict[str, ctk.CTkEntry] = {}

        def add_section(title: str, fields: List[Tuple[str, str]]) -> None:
            card = ctk.CTkFrame(scroll, corner_radius=8)
            card.pack(fill="x", padx=4, pady=(0, 8))
            ctk.CTkLabel(
                card, text=title, font=ctk.CTkFont(size=13, weight="bold")
            ).pack(anchor="w", padx=10, pady=(8, 4))
            grid = ctk.CTkFrame(card, fg_color="transparent")
            grid.pack(fill="x", padx=10, pady=(0, 8))
            grid.grid_columnconfigure(1, weight=1)
            for idx, (key, label) in enumerate(fields):
                ctk.CTkLabel(grid, text=label).grid(row=idx, column=0, sticky="w", pady=2, padx=(0, 8))
                entry = ctk.CTkEntry(grid, width=120)
                entry.grid(row=idx, column=1, sticky="ew", pady=2)
                self._cfg_entries[key] = entry

        add_section(
            "Alturas de Eje Z (mm)",
            [
                ("z_safe_mm", "Z Seguro de traslado:"),
                ("z_pick_mm", "Z para tomar pieza:"),
                ("z_place_mm", "Z para soltar pieza:"),
            ],
        )
        add_section(
            "Posición de Espera (Standby mm)",
            [
                ("standby_x", "Espera X:"),
                ("standby_y", "Espera Y:"),
                ("standby_z", "Espera Z:"),
            ],
        )
        add_section(
            "Reloj de Ajedrez (mm)",
            [
                ("clock_x", "Reloj X:"),
                ("clock_y", "Reloj Y:"),
                ("clock_z", "Reloj Z al presionar:"),
            ],
        )
        add_section(
            "Esquinas Físicas del Tablero en Máquina (mm)",
            [
                ("c0_x", "Esquina A8 (Sup-Izq) X:"),
                ("c0_y", "Esquina A8 (Sup-Izq) Y:"),
                ("c1_x", "Esquina H8 (Sup-Der) X:"),
                ("c1_y", "Esquina H8 (Sup-Der) Y:"),
                ("c2_x", "Esquina H1 (Inf-Der) X:"),
                ("c2_y", "Esquina H1 (Inf-Der) Y:"),
                ("c3_x", "Esquina A1 (Inf-Izq) X:"),
                ("c3_y", "Esquina A1 (Inf-Izq) Y:"),
            ],
        )

        add_section(
            "Audio y Humor (Temporizador)",
            [
                ("delay_warning_seconds", "Demora de aviso humano (seg):"),
            ],
        )

        ctk.CTkButton(
            scroll,
            text="Guardar Configuración",
            fg_color="#2d7d46",
            hover_color="#236337",
            height=36,
            command=self._on_save_config_form,
        ).pack(fill="x", padx=4, pady=8)

    def _build_console_panel(self) -> None:
        console_frame = ctk.CTkFrame(self, corner_radius=8)
        console_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        console_frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(console_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 2))
        ctk.CTkLabel(
            header,
            text="Consola G-code en Vivo",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")

        ctk.CTkButton(
            header,
            text="Limpiar Consola",
            width=115,
            height=24,
            fg_color="#475569",
            command=self._clear_console,
        ).pack(side="right", padx=(4, 0))
        ctk.CTkButton(
            header,
            text="Parámetros ($$)",
            width=115,
            height=24,
            fg_color="#334155",
            command=lambda: self.robot.enqueue_gcode("$$"),
        ).pack(side="right", padx=4)
        ctk.CTkButton(
            header,
            text="Estado (?)",
            width=85,
            height=24,
            fg_color="#334155",
            command=lambda: self.robot.query_status(),
        ).pack(side="right", padx=4)

        self.txt_console = ctk.CTkTextbox(
            console_frame,
            height=135,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.txt_console.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 4))
        self.txt_console.configure(state="disabled")

        input_row = ctk.CTkFrame(console_frame, fg_color="transparent")
        input_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.ent_gcode = ctk.CTkEntry(
            input_row,
            placeholder_text="Ingresá un comando G-code (ej: G0 X50 Y30 Z45, M3 S120, $H)...",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.ent_gcode.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.ent_gcode.bind("<Return>", lambda _e: self._on_send_manual_gcode())

        ctk.CTkButton(
            input_row, text="Enviar", width=95, command=self._on_send_manual_gcode
        ).pack(side="right")

    # ------------------------------------------------------------------
    # Bucle periódico de UI (seguro para hilos)
    # ------------------------------------------------------------------
    def _ui_tick(self) -> None:
        self._drain_queues()
        self._update_video_frame()
        self.after(33, self._ui_tick)

    def _drain_queues(self) -> None:
        while True:
            try:
                direction, message = self._log_queue.get_nowait()
                self._log_to_console(direction, message)
            except queue.Empty:
                break

        latest_status: Optional[MachineStatus] = None
        while True:
            try:
                latest_status = self._status_queue.get_nowait()
            except queue.Empty:
                break
        if latest_status is not None:
            self._render_machine_status(latest_status)

        while True:
            try:
                connected, port = self._conn_queue.get_nowait()
                self._render_connection_state(connected, port)
            except queue.Empty:
                break

    def _render_machine_status(self, st: MachineStatus) -> None:
        state_colors = {
            "Idle": "#22c55e",
            "Run": "#38bdf8",
            "Jog": "#38bdf8",
            "Home": "#a855f7",
            "Hold": "#f59e0b",
            "Hold:0": "#f59e0b",
            "Hold:1": "#f59e0b",
            "Alarm": "#ef4444",
            "Desconectado": "#f59e0b",
        }
        color = state_colors.get(st.state, "#e2e8f0")
        sim_suffix = " (SIM)" if st.simulated else ""
        self.lbl_grbl_state.configure(text=f"Estado: {st.state}{sim_suffix}", text_color=color)
        self.lbl_grbl_pos.configure(text=f"X: {st.x:6.2f}  Y: {st.y:6.2f}  Z: {st.z:6.2f} mm")

    def _render_connection_state(self, connected: bool, port: str) -> None:
        is_sim = port == "SIMULACIÓN" or self.robot.is_simulation
        if connected:
            if is_sim:
                self._sim_mode_var.set(True)
                self.btn_connect.configure(text="Conectar", state="disabled", fg_color="#475569")
                self.com_combo.configure(state="disabled")
            else:
                self._sim_mode_var.set(False)
                self.btn_connect.configure(
                    text="Desconectar", state="normal", fg_color="#b91c1c", hover_color="#991b1b"
                )
                self.com_combo.configure(state="readonly")
                self.config_manager.set("serial", "port", value=port)
            self.audio.announce("connected")
        else:
            self._sim_mode_var.set(False)
            self.btn_connect.configure(
                text="Conectar", state="normal", fg_color="#1f6aa5", hover_color="#144870"
            )
            self.com_combo.configure(state="readonly")
            self.audio.announce("disconnected")

    def _log_to_console(self, direction: str, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        prefix_map = {
            "TX": ">> TX",
            "RX": "<< RX",
            "SYS": "[SYS]",
            "ERR": "[ERR]",
            "SIM": "[SIM]",
        }
        prefix = prefix_map.get(direction, f"[{direction}]")
        line = f"[{ts}] {prefix} {message}\n"
        self.txt_console.configure(state="normal")
        self.txt_console.insert("end", line)
        # Limita historial para mantener fluidez de memoria
        if int(self.txt_console.index("end-1c").split(".")[0]) > 500:
            self.txt_console.delete("1.0", "100.0")
        self.txt_console.see("end")
        self.txt_console.configure(state="disabled")

    def _clear_console(self) -> None:
        self.txt_console.configure(state="normal")
        self.txt_console.delete("1.0", "end")
        self.txt_console.configure(state="disabled")

    # ------------------------------------------------------------------
    # Video y Calibración de 4 Esquinas
    # ------------------------------------------------------------------
    def _try_autostart_camera(self) -> None:
        idx = int(self._camera_idx_var.get())
        if self.vision.start(idx):
            self.btn_camera.configure(text="Detener Cámara", fg_color="#b91c1c")
            self._log_to_console("SYS", f"Cámara índice {idx} iniciada correctamente.")
        else:
            self._log_to_console("SYS", f"No se detectó cámara automática en índice {idx}.")

    def _toggle_camera(self) -> None:
        if self.vision.is_running:
            self.vision.stop()
            self.btn_camera.configure(text="Iniciar Cámara", fg_color="#1f6aa5")
            self.video_label.configure(
                image="",
                text="Cámara detenida.\nSeleccioná un índice y hacé clic en 'Iniciar Cámara'.",
            )
            self._log_to_console("SYS", "Cámara detenida.")
        else:
            idx = int(self._camera_idx_var.get())
            if self.vision.start(idx):
                self.config_manager.set("vision", "camera_index", value=idx)
                self.btn_camera.configure(text="Detener Cámara", fg_color="#b91c1c")
                self._log_to_console("SYS", f"Cámara índice {idx} iniciada.")
            else:
                messagebox.showwarning(
                    "Cámara no disponible",
                    f"No se pudo abrir la cámara índice {idx}.\nVerificá que la webcam esté conectada y libre.",
                )

    def _update_video_frame(self) -> None:
        if not self.vision.is_running:
            return
        frame_bgr, detections = self.vision.get_latest_state()
        if frame_bgr is None:
            return

        self._last_detections = detections
        fh, fw = frame_bgr.shape[:2]
        self._last_frame_shape = (fw, fh)

        rendered_bgr = self.vision.render_overlay(
            frame_bgr,
            detections,
            show_grid=self._show_grid_var.get(),
            show_aruco=self._show_aruco_var.get(),
            pending_calibration_points=self._pending_calib_pts if self._calibrating else None,
        )

        # Ajuste proporcional al tamaño real del visor de video (mapeo 1:1 para clics de calibración)
        cw = max(320, self.video_label.winfo_width())
        ch = max(240, self.video_label.winfo_height())
        scale = min(cw / fw, ch / fh)
        new_w = max(1, int(fw * scale))
        new_h = max(1, int(fh * scale))
        offset_x = (cw - new_w) / 2.0
        offset_y = (ch - new_h) / 2.0
        self._display_transform = (scale, scale, offset_x, offset_y)

        rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Crea lienzo con fondo oscuro exacto del visor para que las coordenadas del clic coincidan
        letterboxed = Image.new("RGB", (cw, ch), (15, 17, 23))
        letterboxed.paste(pil_img, (int(offset_x), int(offset_y)))

        self._video_photo = ImageTk.PhotoImage(letterboxed)
        self.video_label.configure(image=self._video_photo, text="")

        valid_pieces = sum(1 for d in detections if d.piece is not None)
        calib_txt = "Activa" if self.vision.calibrator.is_calibrated else "Pendiente"
        self.lbl_vision_stats.configure(
            text=f"Piezas detectadas: {valid_pieces} / 32  |  FPS: {self.vision.fps:.1f}  |  Calibración: {calib_txt}"
        )

    def _toggle_calibration_mode(self) -> None:
        if not self.vision.is_running:
            messagebox.showinfo(
                "Iniciá la cámara primero",
                "Para calibrar el tablero primero debés iniciar la cámara cenital.",
            )
            return

        if self._calibrating:
            self._calibrating = False
            self._pending_calib_pts.clear()
            self.btn_calibrate.configure(text="Calibrar Tablero", fg_color="#0e7490")
            self.lbl_calib_banner.configure(
                text="Calibración cancelada.", fg_color="#1f2937"
            )
        else:
            self._calibrating = True
            self._pending_calib_pts.clear()
            self.btn_calibrate.configure(text="Cancelar Calibración", fg_color="#b45309")
            self.lbl_calib_banner.configure(
                text=CORNER_INSTRUCTIONS_ES[0], fg_color="#0e7490"
            )
            self._log_to_console("SYS", "Asistente de calibración iniciado (A8 -> H8 -> H1 -> A1).")

    def _on_video_click(self, event: tk.Event) -> None:
        if not self._calibrating or self._last_frame_shape is None:
            return

        sx, sy, ox, oy = self._display_transform
        fw, fh = self._last_frame_shape
        fx = (event.x - ox) / sx
        fy = (event.y - oy) / sy

        if not (0 <= fx <= fw and 0 <= fy <= fh):
            return

        self._pending_calib_pts.append((float(fx), float(fy)))
        step = len(self._pending_calib_pts)
        self._log_to_console("SYS", f"Esquina {step}/4 marcada en píxel ({fx:.1f}, {fy:.1f}).")

        if step < 4:
            self.lbl_calib_banner.configure(text=CORNER_INSTRUCTIONS_ES[step])
            return

        # 4 esquinas completas: calcula homografía y guarda en config.json
        pts_list = [[round(x, 2), round(y, 2)] for x, y in self._pending_calib_pts]
        self.vision.calibrator.set_pixel_corners(pts_list)
        self.config_manager.set("vision", "calibration", "pixel_corners", value=pts_list, autosave=False)
        self.config_manager.set("vision", "calibration", "calibrated", value=True, autosave=True)

        self._calibrating = False
        self._pending_calib_pts.clear()
        self.btn_calibrate.configure(text="Calibrar Tablero", fg_color="#0e7490")
        self.lbl_calib_banner.configure(
            text="¡Tablero calibrado correctamente! Homografía píxeles <-> 8x8 <-> mm activa.",
            fg_color="#166534",
        )
        self._log_to_console("SYS", "Calibración de 4 esquinas guardada en config.json.")
        self.audio.announce("calibration_saved")

    def _clear_calibration(self) -> None:
        self._calibrating = False
        self._pending_calib_pts.clear()
        self.vision.calibrator.clear_pixel_calibration()
        self.config_manager.set("vision", "calibration", "pixel_corners", value=[], autosave=False)
        self.config_manager.set("vision", "calibration", "calibrated", value=False, autosave=True)
        self.btn_calibrate.configure(text="Calibrar Tablero", fg_color="#0e7490")
        self.lbl_calib_banner.configure(
            text="Calibración eliminada. Presioná 'Calibrar Tablero' para definir las 4 esquinas.",
            fg_color="#1f2937",
        )
        self._log_to_console("SYS", "Calibración de tablero borrada.")

    # ------------------------------------------------------------------
    # Acciones de Barra Serial y Robot
    # ------------------------------------------------------------------
    def _refresh_com_ports(self) -> None:
        ports = self.robot.list_com_ports()
        self._com_display_to_device.clear()
        if not ports:
            self.com_combo.configure(values=["Sin puertos COM"])
            self.com_combo.set("Sin puertos COM")
            return

        labels = []
        saved_port = str(self.config_manager.get("serial", "port", default=""))
        selected_label = None
        for dev, desc in ports:
            self._com_display_to_device[desc] = dev
            labels.append(desc)
            if dev == saved_port:
                selected_label = desc

        self.com_combo.configure(values=labels)
        self.com_combo.set(selected_label or labels[0])

    def _toggle_connection(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()
            return
        selected = self.com_combo.get()
        device = self._com_display_to_device.get(selected, "")
        if not device:
            messagebox.showwarning(
                "Sin puerto seleccionado",
                "Conectá el Arduino Uno por USB y presioná 'Actualizar', o activá 'Modo Simulación'.",
            )
            return
        baud = int(self.config_manager.get("serial", "baudrate", default=115200))
        ok, msg = self.robot.connect(device, baudrate=baud)
        if not ok:
            messagebox.showerror("Error de conexión", msg)

    def _on_toggle_simulation_mode(self) -> None:
        if self._sim_mode_var.get():
            self.robot.connect_simulation()
        else:
            if self.robot.is_simulation:
                self.robot.disconnect()

    def _on_homing(self) -> None:
        if not self._require_connected():
            return
        self.audio.announce("homing_start")
        self.robot.home()

    def _on_unlock(self) -> None:
        if not self._require_connected():
            return
        self.robot.unlock()

    def _on_emergency_stop(self) -> None:
        self.robot.emergency_stop()
        self.audio.announce("emergency_stop", interrupt=True)

    def _on_soft_reset(self) -> None:
        if not self._require_connected():
            return
        self.robot.soft_reset()

    def _require_connected(self) -> bool:
        if not self.robot.is_connected:
            messagebox.showwarning(
                "Controlador desconectado",
                "Primero conectá el Arduino o activá el 'Modo Simulación' en la barra superior.",
            )
            return False
        return True

    def _jog(self, sx: int, sy: int, sz: int) -> None:
        if not self._require_connected():
            return
        step = float(self._jog_step_mm.get())
        self.robot.jog_relative(dx=sx * step, dy=sy * step, dz=sz * step)

    def _on_go_standby(self) -> None:
        if not self._require_connected():
            return
        self.robot.go_to_standby()

    def _on_go_safe_z(self) -> None:
        if not self._require_connected():
            return
        self.robot.go_to_safe_z()

    def _on_move_simultaneous(self) -> None:
        if not self._require_connected():
            return
        try:
            x = float(self.ent_sim_x.get().strip())
            y = float(self.ent_sim_y.get().strip())
            z = float(self.ent_sim_z.get().strip())
        except ValueError:
            messagebox.showerror("Valores inválidos", "Ingresá números válidos para X, Y y Z.")
            return
        self.robot.move_simultaneous(x, y, z, rapid=True)

    # ------------------------------------------------------------------
    # Garra, Reloj, Perfiles GRBL y Voz
    # ------------------------------------------------------------------
    def _on_gripper_slider_drag(self, value: float) -> None:
        val = int(round(value))
        self._gripper_pwm_var.set(val)
        self.lbl_gripper_val.configure(text=f"S{val}")

    def _on_send_gripper_slider(self) -> None:
        if not self._require_connected():
            return
        self.robot.set_gripper_pwm(self._gripper_pwm_var.get(), dwell=False)

    def _on_open_gripper(self) -> None:
        val = int(self.config_manager.get("gripper", "open_pwm", default=65))
        self.slider_gripper.set(val)
        self._on_gripper_slider_drag(val)
        if not self._require_connected():
            return
        self.robot.open_gripper()

    def _on_close_gripper(self) -> None:
        val = int(self.config_manager.get("gripper", "close_pwm", default=155))
        self.slider_gripper.set(val)
        self._on_gripper_slider_drag(val)
        if not self._require_connected():
            return
        self.robot.close_gripper()

    def _on_test_gripper(self) -> None:
        if not self._require_connected():
            return
        self.robot.test_gripper_cycle()

    def _save_gripper_preset(self, key: str) -> None:
        val = int(self._gripper_pwm_var.get())
        self.config_manager.set("gripper", key, value=val)
        etiqueta = "Abierta" if key == "open_pwm" else "Cerrada"
        self._log_to_console("SYS", f"Valor de garra '{etiqueta}' guardado en config.json: M3 S{val}.")

    def _on_save_clock_from_current(self) -> None:
        if not self._require_connected():
            return
        st = self.robot.status
        self.config_manager.set("chess_clock", "x", value=round(st.x, 2), autosave=False)
        self.config_manager.set("chess_clock", "y", value=round(st.y, 2), autosave=False)
        self.config_manager.set("chess_clock", "z_press", value=round(st.z, 2), autosave=True)
        self._sync_config_fields_to_ui()
        self._log_to_console(
            "SYS",
            f"Coordenada del reloj guardada desde posición actual: X={st.x:.2f} Y={st.y:.2f} Z={st.z:.2f}.",
        )

    def _on_press_clock(self) -> None:
        if not self._require_connected():
            return
        self.robot.press_chess_clock(return_standby=True)
        self.audio.announce("clock_pressed")

    def _on_apply_grbl_preset(self) -> None:
        label = self.preset_combo.get()
        key = self._preset_label_to_key.get(label, "normal")
        ok, msg = self.robot.apply_grbl_preset(key)
        if ok:
            self.config_manager.set("grbl_presets", "active_preset", value=key)
            self._log_to_console("SYS", msg)
        else:
            messagebox.showwarning("Perfil GRBL", msg)

    def _on_test_voice(self) -> None:
        selected = self.voice_combo.get()
        for label, key in VOICE_TEST_LABELS_ES:
            if label == selected:
                self.audio.announce(key, interrupt=True)
                self._log_to_console("SYS", f"Probando voz (es-AR): '{EVENT_PHRASES_ES_AR[key]}'")
                return

    def _on_test_humor_category(self, category: str) -> None:
        res = self.audio.play_event(category)
        if res is None:
            self._log_to_console("ERR", f"No hay sonidos ni frases configuradas para '{category}'.")

    def _on_toggle_humor_timer(self) -> None:
        enabled = bool(self._humor_timer_var.get())
        self.audio.set_delay_timer_enabled(enabled)
        self.config_manager.set("audio", "humor_timer_enabled", value=enabled)
        estado = "activado" if enabled else "desactivado"
        self._log_to_console("SYS", f"Temporizador de demora {estado}.")

    def _sync_human_delay_timer(self) -> None:
        """Activa el temporizador de demora solo cuando es el turno del jugador humano."""
        if self.chess_engine.board.is_game_over():
            self.audio.stop_delay_timer()
            return
        robot_color = str(self.config_manager.get("chess", "robot_color", default="black")).lower()
        human_is_white = robot_color != "white"
        is_human_turn = (self.chess_engine.board.turn is True) if human_is_white else (self.chess_engine.board.turn is False)
        if is_human_turn:
            self.audio.start_delay_timer()
        else:
            self.audio.stop_delay_timer()

    def _trigger_humor_for_outcome(self, outcome: Any) -> None:
        if "checkmate" in outcome.audio_events:
            self.audio.play_event("victoria")
        elif "check" in outcome.audio_events:
            self.audio.play_event("jaque")
        elif outcome.is_capture:
            self.audio.play_event("captura")
        else:
            for ev in outcome.audio_events:
                self.audio.announce(ev)

    # ------------------------------------------------------------------
    # Tablero, Ajedrez y Consola Manual
    # ------------------------------------------------------------------
    def _update_chess_panel(self) -> None:
        self.lbl_game_status.configure(text=self.chess_engine.get_status_text_es())
        self.txt_board_ascii.configure(state="normal")
        self.txt_board_ascii.delete("1.0", "end")
        self.txt_board_ascii.insert("1.0", str(self.chess_engine.board))
        self.txt_board_ascii.configure(state="disabled")

    def _on_execute_manual_move(self) -> None:
        move_str = self.ent_manual_move.get().strip()
        outcome = self.chess_engine.apply_move_str(move_str)
        self._log_to_console("SYS", outcome.message_es)

        if not outcome.valid or not outcome.move_uci:
            for ev in outcome.audio_events:
                self.audio.announce(ev)
            messagebox.showwarning("Jugada inválida", outcome.message_es)
            return

        self._trigger_humor_for_outcome(outcome)
        self.ent_manual_move.delete(0, "end")
        self._update_chess_panel()
        self._sync_human_delay_timer()

        if not self.robot.is_connected:
            self._log_to_console("SYS", "Jugada registrada lógicamente (robot desconectado).")
            return

        src_sq = outcome.move_uci[:2]
        dst_sq = outcome.move_uci[2:4]
        src_xy = self.vision.calibrator.square_to_machine_mm(src_sq)
        dst_xy = self.vision.calibrator.square_to_machine_mm(dst_sq)

        if outcome.is_castling and outcome.rook_castling_move:
            r_src, r_dst = outcome.rook_castling_move
            self.robot.execute_pick_and_place(src_xy, dst_xy, is_capture=False, press_clock_after=False)
            self.robot.execute_pick_and_place(
                self.vision.calibrator.square_to_machine_mm(r_src),
                self.vision.calibrator.square_to_machine_mm(r_dst),
                is_capture=False,
                press_clock_after=True,
            )
        else:
            self.robot.execute_pick_and_place(src_xy, dst_xy, is_capture=outcome.is_capture)

    def _on_detect_move_from_vision(self) -> None:
        if not self.vision.calibrator.is_calibrated:
            messagebox.showinfo(
                "Falta calibración",
                "Calibrá primero las 4 esquinas del tablero para mapear los ArUco a casillas.",
            )
            return
        observed = self.vision.get_board_occupancy_map()
        outcome = self.chess_engine.detect_human_move_from_vision(observed)
        self._log_to_console("SYS", outcome.message_es)
        if outcome.valid:
            self._trigger_humor_for_outcome(outcome)
            self._update_chess_panel()
            self._sync_human_delay_timer()
        else:
            for ev in outcome.audio_events:
                self.audio.announce(ev)

    def _on_reset_game(self) -> None:
        self.chess_engine.reset_game()
        self._update_chess_panel()
        self._log_to_console("SYS", "Partida reiniciada a posición inicial.")
        self.audio.play_event("inicio")
        self._sync_human_delay_timer()

    def _on_send_manual_gcode(self) -> None:
        cmd = self.ent_gcode.get().strip()
        if not cmd:
            return
        self.ent_gcode.delete(0, "end")
        self.robot.enqueue_gcode(cmd)

    # ------------------------------------------------------------------
    # Sincronización y guardado de Configuración
    # ------------------------------------------------------------------
    def _sync_config_fields_to_ui(self) -> None:
        cfg = self.config_manager.data
        motion = cfg["motion"]
        clock = cfg["chess_clock"]
        audio_cfg = cfg["audio"]
        corners = cfg["vision"]["calibration"]["machine_corners_mm"]

        self.lbl_clock_coords.configure(
            text=f"Coordenada guardada: X={clock['x']:.1f}  Y={clock['y']:.1f}  Z={clock['z_press']:.1f} mm"
        )
        delay_s = int(round(float(audio_cfg.get("delay_warning_seconds", 30))))
        if hasattr(self, "sw_humor_timer"):
            self.sw_humor_timer.configure(text=f"Activar temporizador de demora ({delay_s}s)")

        values = {
            "z_safe_mm": motion["z_safe_mm"],
            "z_pick_mm": motion["z_pick_mm"],
            "z_place_mm": motion["z_place_mm"],
            "standby_x": motion["standby_position"]["x"],
            "standby_y": motion["standby_position"]["y"],
            "standby_z": motion["standby_position"]["z"],
            "clock_x": clock["x"],
            "clock_y": clock["y"],
            "clock_z": clock["z_press"],
            "c0_x": corners[0][0],
            "c0_y": corners[0][1],
            "c1_x": corners[1][0],
            "c1_y": corners[1][1],
            "c2_x": corners[2][0],
            "c2_y": corners[2][1],
            "c3_x": corners[3][0],
            "c3_y": corners[3][1],
            "delay_warning_seconds": audio_cfg.get("delay_warning_seconds", 30),
        }
        for key, val in values.items():
            ent = self._cfg_entries.get(key)
            if ent is not None:
                ent.delete(0, "end")
                ent.insert(0, f"{float(val):.2f}")

    def _on_save_config_form(self) -> None:
        try:
            vals = {k: float(ent.get().strip()) for k, ent in self._cfg_entries.items()}
        except ValueError:
            messagebox.showerror(
                "Error de formato",
                "Todos los parámetros de configuración deben ser números válidos (usá punto decimal).",
            )
            return

        cm = self.config_manager
        cm.set("motion", "z_safe_mm", value=vals["z_safe_mm"], autosave=False)
        cm.set("motion", "z_pick_mm", value=vals["z_pick_mm"], autosave=False)
        cm.set("motion", "z_place_mm", value=vals["z_place_mm"], autosave=False)
        cm.set(
            "motion",
            "standby_position",
            value={"x": vals["standby_x"], "y": vals["standby_y"], "z": vals["standby_z"]},
            autosave=False,
        )
        cm.set("chess_clock", "x", value=vals["clock_x"], autosave=False)
        cm.set("chess_clock", "y", value=vals["clock_y"], autosave=False)
        cm.set("chess_clock", "z_press", value=vals["clock_z"], autosave=False)

        corners_mm = [
            [vals["c0_x"], vals["c0_y"]],
            [vals["c1_x"], vals["c1_y"]],
            [vals["c2_x"], vals["c2_y"]],
            [vals["c3_x"], vals["c3_y"]],
        ]
        cm.set("vision", "calibration", "machine_corners_mm", value=corners_mm, autosave=False)
        delay_sec = max(3.0, vals["delay_warning_seconds"])
        cm.set("audio", "delay_warning_seconds", value=delay_sec, autosave=True)
        self.audio.set_delay_warning_seconds(delay_sec)
        self.vision.calibrator.set_machine_corners_mm(corners_mm)
        self._sync_config_fields_to_ui()
        self._log_to_console("SYS", "Configuración guardada y homografía en milímetros actualizada.")
        messagebox.showinfo("Configuración guardada", "Los parámetros se guardaron correctamente en config.json.")

    # ------------------------------------------------------------------
    # Cierre limpio
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        try:
            self.vision.stop()
        except Exception:
            pass
        try:
            self.robot.disconnect()
        except Exception:
            pass
        try:
            self.chess_engine.stop_uci_engine()
        except Exception:
            pass
        try:
            self.audio.shutdown()
        except Exception:
            pass
        self.destroy()
