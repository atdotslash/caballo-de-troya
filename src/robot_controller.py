"""Controlador serial no bloqueante para Arduino Uno + CNC Shield V3 (grbl-servo).

Características principales:
- Comunicación asíncrona en hilo dedicado con espera real de respuestas `ok` / `error:` / `ALARM:`.
- Modo Simulación integrado sin puerto COM: simula retardos reales (100–300 ms),
  actualiza coordenadas MPos (X, Y, Z) en tiempo real y registra todo con prefijo `[SIM]`.
- Comandos en tiempo real de GRBL sin espera de cola: `!` (Parada de emergencia), `~` (Reanudar),
  `Ctrl-X` (Reset suave) y `?` (Consulta de estado y coordenadas MPos/WPos).
- Movimiento fluido simultáneo en 3 ejes: cada traslado combina X, Y y Z en una sola línea G-code
  (`G0 X.. Y.. Z..` o `G1 X.. Y.. Z.. F..`).
- Control de garra SG90 en pin D11 (`M3 S<val>` y `M5`) con pausa de asentamiento (`G4 P<seg>`).
- Rutina automática de pulsación del reloj de ajedrez y retorno a posición de espera.
"""

from __future__ import annotations

import logging
import queue
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)

STATUS_REGEX = re.compile(
    r"^<(?P<state>[^|>]+)(?:\|(?P<fields>[^>]+))?>$"
)
POS_REGEX = re.compile(
    r"(?:MPos|WPos):(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)"
)
AXIS_TOKEN_REGEX = re.compile(r"([XYZPS])(-?\d+(?:\.\d+)?)", re.IGNORECASE)
GRBL_PARAM_REGEX = re.compile(r"^(\$\d+)=(-?\d+(?:\.\d+)?)$")


def _fmt_gcode_num(value: float) -> str:
    """Formatea coordenadas de jog limpiamente (ej: -10.0 -> '-10.0', 0.5 -> '0.5')."""
    text = f"{value:.3f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


@dataclass
class MachineStatus:
    """Último estado reportado por GRBL (real o simulado) mediante consulta `?`."""

    state: str = "Desconectado"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    raw: str = ""
    simulated: bool = False


@dataclass
class _QueuedCommand:
    gcode: str
    timeout_s: float
    done_event: threading.Event
    success: bool = False
    response: str = ""


class RobotController:
    """Controlador de alto y bajo nivel para el brazo SCARA con firmware grbl-servo."""

    def __init__(
        self,
        config_getter: Callable[[], Dict[str, Any]],
        on_log: Optional[Callable[[str, str], None]] = None,
        on_status: Optional[Callable[[MachineStatus], None]] = None,
        on_connection_change: Optional[Callable[[bool, str], None]] = None,
        on_grbl_settings: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        self._get_config = config_getter
        self._on_log = on_log
        self._on_status = on_status
        self._on_connection_change = on_connection_change
        self._on_grbl_settings = on_grbl_settings

        self._serial: Optional[serial.Serial] = None
        self._serial_lock = threading.RLock()
        self._connected = False
        self._simulation_mode = False
        self._port_name = ""

        self._cmd_queue: queue.Queue[_QueuedCommand] = queue.Queue()
        self._pending_cmd_lock = threading.Lock()
        self._active_cmd: Optional[_QueuedCommand] = None

        self._stop_event = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._sim_thread: Optional[threading.Thread] = None

        # Caché de parámetros EEPROM leídos desde GRBL ($100, $101, $102, $110..$122, etc.)
        self._grbl_settings_lock = threading.RLock()
        self._grbl_settings: Dict[str, float] = {}

        # Estado interno del simulador mock
        self._sim_lock = threading.RLock()
        self._sim_x = 0.0
        self._sim_y = 0.0
        self._sim_z = 0.0
        self._sim_abs_mode = True
        self._sim_state = "Idle"
        self._sim_pwm = 0
        steps_cfg = self._get_config().get("motion", {}).get("steps_per_mm", {})
        self._sim_params: Dict[str, float] = {
            "$100": float(steps_cfg.get("$100", 80.0)),
            "$101": float(steps_cfg.get("$101", 80.0)),
            "$102": float(steps_cfg.get("$102", 400.0)),
            "$110": 3000.0,
            "$111": 3000.0,
            "$112": 1500.0,
            "$120": 200.0,
            "$121": 200.0,
            "$122": 120.0,
        }

        self._status = MachineStatus()
        self._capture_count = 0

    @staticmethod
    def list_com_ports() -> List[Tuple[str, str]]:
        """Detecta puertos COM disponibles en Windows (device, descripción)."""
        ports = []
        for info in serial.tools.list_ports.comports():
            desc = info.description or "Puerto serie"
            ports.append((info.device, f"{info.device} — {desc}"))
        return ports

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_simulation(self) -> bool:
        return self._simulation_mode

    @property
    def port_name(self) -> str:
        return self._port_name

    @property
    def status(self) -> MachineStatus:
        return self._status

    def _emit_log(self, direction: str, message: str) -> None:
        logger.debug("[%s] %s", direction, message)
        if self._on_log:
            try:
                self._on_log(direction, message)
            except Exception:
                pass

    def _emit_status(self, status: MachineStatus) -> None:
        self._status = status
        if self._on_status:
            try:
                self._on_status(status)
            except Exception:
                pass

    def _emit_connection(self, connected: bool, port: str) -> None:
        if self._on_connection_change:
            try:
                self._on_connection_change(connected, port)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Conexión real (COM) y Modo Simulación (sin hardware)
    # ------------------------------------------------------------------
    def connect(self, port: str, baudrate: int = 115200) -> Tuple[bool, str]:
        """Abre el puerto COM del Arduino e inicia los hilos de lectura/envío GRBL."""
        if self._connected:
            self.disconnect()

        if not port:
            return False, "No se seleccionó ningún puerto COM."

        try:
            ser = serial.Serial(
                port=port,
                baudrate=int(baudrate),
                timeout=0.15,
                write_timeout=1.0,
            )
        except Exception as exc:
            msg = f"No se pudo abrir {port}: {exc}"
            self._emit_log("ERR", msg)
            return False, msg

        self._serial = ser
        self._port_name = port
        self._simulation_mode = False
        self._connected = True
        self._stop_event.clear()
        self._clear_queue()

        self._emit_log("SYS", f"Conectado a {port} ({baudrate} baudios). Esperando inicialización de GRBL...")
        self._emit_status(MachineStatus(state="Inicializando", simulated=False))
        self._emit_connection(True, port)

        self._rx_thread = threading.Thread(target=self._rx_loop, name="GRBL-RX", daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, name="GRBL-TX", daemon=True)
        self._poll_thread = threading.Thread(target=self._poll_loop, name="GRBL-Poll", daemon=True)
        self._rx_thread.start()
        self._tx_thread.start()
        self._poll_thread.start()

        threading.Thread(target=self._post_connect_init, name="GRBL-Init", daemon=True).start()
        return True, f"Conectado a {port}"

    def connect_simulation(self) -> Tuple[bool, str]:
        """Activa el Modo Simulación sin requerir puerto COM físico."""
        if self._connected:
            self.disconnect()

        with self._sim_lock:
            self._sim_x = 0.0
            self._sim_y = 0.0
            self._sim_z = 0.0
            self._sim_abs_mode = True
            self._sim_state = "Idle"
            self._sim_pwm = int(self._get_config().get("gripper", {}).get("open_pwm", 65))

        self._serial = None
        self._port_name = "SIMULACIÓN"
        self._simulation_mode = True
        self._connected = True
        self._stop_event.clear()
        self._clear_queue()

        self._emit_log("SIM", "Modo Simulación activado — Grbl 1.1h (grbl-servo virtual) listo.")
        self._emit_sim_status()
        self._emit_connection(True, "SIMULACIÓN")

        self._sim_thread = threading.Thread(
            target=self._sim_tx_loop, name="GRBL-SimWorker", daemon=True
        )
        self._sim_thread.start()
        return True, "Modo Simulación activado"

    def _emit_sim_status(self) -> None:
        with self._sim_lock:
            raw = f"<{self._sim_state}|MPos:{self._sim_x:.3f},{self._sim_y:.3f},{self._sim_z:.3f}|FS:0,0>"
            st = MachineStatus(
                state=self._sim_state,
                x=self._sim_x,
                y=self._sim_y,
                z=self._sim_z,
                raw=raw,
                simulated=True,
            )
        self._emit_status(st)

    def _post_connect_init(self) -> None:
        """Espera el reinicio del bootloader de Arduino Uno y configura milímetros + absolutas."""
        time.sleep(2.0)
        if not self._connected or self._simulation_mode:
            return
        try:
            with self._serial_lock:
                if self._serial and self._serial.is_open:
                    self._serial.write(b"\r\n\r\n")
                    self._serial.flush()
            time.sleep(0.2)
            self.enqueue_gcode("G21")
            self.enqueue_gcode("G90")
            self.query_status()
        except Exception as exc:
            self._emit_log("ERR", f"Error en secuencia inicial GRBL: {exc}")

    def disconnect(self) -> None:
        """Cierra el puerto serial o desactiva el Modo Simulación y detiene los hilos limpiamente."""
        was_connected = self._connected
        was_sim = self._simulation_mode
        old_port = self._port_name

        self._connected = False
        self._simulation_mode = False
        self._stop_event.set()
        self._clear_queue()

        with self._pending_cmd_lock:
            if self._active_cmd is not None:
                self._active_cmd.success = False
                self._active_cmd.response = "Desconectado"
                self._active_cmd.done_event.set()
                self._active_cmd = None

        with self._serial_lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None

        for th in (self._tx_thread, self._rx_thread, self._poll_thread, self._sim_thread):
            if th and th.is_alive() and th is not threading.current_thread():
                th.join(timeout=1.0)

        self._port_name = ""
        self._emit_status(MachineStatus(state="Desconectado", simulated=False))
        if was_connected:
            if was_sim:
                self._emit_log("SIM", "Modo Simulación desactivado.")
            else:
                self._emit_log("SYS", f"Puerto {old_port} desconectado.")
            self._emit_connection(False, "")

    def _clear_queue(self) -> None:
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
                cmd.success = False
                cmd.response = "Cancelado"
                cmd.done_event.set()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Hilo simulador mock de GRBL (100–300 ms, MPos en tiempo real, [SIM])
    # ------------------------------------------------------------------
    def _sim_tx_loop(self) -> None:
        while not self._stop_event.is_set() and self._connected and self._simulation_mode:
            try:
                cmd = self._cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._connected or not self._simulation_mode:
                cmd.success = False
                cmd.response = "Simulación detenida"
                cmd.done_event.set()
                continue

            line = cmd.gcode.strip()
            self._emit_log("SIM", f">> {line}")
            ok, response = self._execute_simulated_command(line)
            cmd.success = ok
            cmd.response = response
            cmd.done_event.set()
            self._emit_log("SIM", f"<< {response}")

    def _execute_simulated_command(self, line: str) -> Tuple[bool, str]:
        upper = line.upper().strip()

        # Si está en Hold por parada de emergencia, rechaza movimientos hasta Reset o Resume
        with self._sim_lock:
            if self._sim_state == "Hold" and upper not in ("$X", "$$"):
                return False, "error:9 (Bloqueado en Hold — usá Reiniciar o Reanudar)"

        # Homing físico simulado ($H)
        if upper == "$H":
            with self._sim_lock:
                self._sim_state = "Home"
            self._emit_sim_status()
            time.sleep(random.uniform(0.22, 0.30))
            with self._sim_lock:
                self._sim_x = 0.0
                self._sim_y = 0.0
                self._sim_z = 0.0
                self._sim_state = "Idle"
            self._emit_sim_status()
            return True, "ok"

        # Desbloqueo ($X)
        if upper == "$X":
            time.sleep(0.10)
            with self._sim_lock:
                self._sim_state = "Idle"
            self._emit_sim_status()
            return True, "ok"

        # Consulta de parámetros EEPROM ($$)
        if upper == "$$":
            time.sleep(0.12)
            with self._sim_lock:
                snapshot = dict(self._sim_params)
            for k, v in snapshot.items():
                self._emit_log("SIM", f"<< {k}={v:.3f}")
                self._record_grbl_setting(k, v)
            return True, "ok"

        # Escritura de parámetro GRBL ($100=80.0, $110=3000.0, etc.)
        param_match = GRBL_PARAM_REGEX.match(upper)
        if param_match:
            time.sleep(0.10)
            key = param_match.group(1)
            val = float(param_match.group(2))
            with self._sim_lock:
                self._sim_params[key] = val
            self._record_grbl_setting(key, val)
            return True, "ok"

        # Modo absoluto / relativo / unidades
        if upper in ("G21", "G20", "G90", "G91"):
            time.sleep(0.10)
            with self._sim_lock:
                if upper == "G90":
                    self._sim_abs_mode = True
                elif upper == "G91":
                    self._sim_abs_mode = False
            return True, "ok"

        # Pausa sincronizada (G4 P<seg>)
        if upper.startswith("G4"):
            tokens = {k.upper(): float(v) for k, v in AXIS_TOKEN_REGEX.findall(upper)}
            dwell_s = min(0.30, max(0.10, tokens.get("P", 0.15)))
            time.sleep(dwell_s)
            return True, "ok"

        # Servo garra SG90 (M3 S<val> / M5)
        if upper.startswith("M3"):
            tokens = {k.upper(): float(v) for k, v in AXIS_TOKEN_REGEX.findall(upper)}
            pwm = int(round(tokens.get("S", 0)))
            time.sleep(random.uniform(0.10, 0.16))
            with self._sim_lock:
                self._sim_pwm = pwm
            self._emit_log("SIM", f"Garra SG90 posicionada en PWM S{pwm}.")
            return True, "ok"

        if upper.startswith("M5"):
            time.sleep(0.10)
            with self._sim_lock:
                self._sim_pwm = 0
            self._emit_log("SIM", "Garra SG90 PWM apagado (M5).")
            return True, "ok"

        # Jog GRBL 1.1 ($J=G91 G21 X.. Y.. Z.. F..)
        if upper.startswith("$J="):
            is_rel = "G91" in upper
            tokens = {k.upper(): float(v) for k, v in AXIS_TOKEN_REGEX.findall(upper)}
            with self._sim_lock:
                self._sim_state = "Jog"
            self._emit_sim_status()
            time.sleep(random.uniform(0.12, 0.24))
            with self._sim_lock:
                if is_rel:
                    self._sim_x += tokens.get("X", 0.0)
                    self._sim_y += tokens.get("Y", 0.0)
                    self._sim_z += tokens.get("Z", 0.0)
                else:
                    if "X" in tokens:
                        self._sim_x = tokens["X"]
                    if "Y" in tokens:
                        self._sim_y = tokens["Y"]
                    if "Z" in tokens:
                        self._sim_z = tokens["Z"]
                self._sim_state = "Idle"
            self._emit_sim_status()
            return True, "ok"

        # Movimientos lineales / rápidos simultáneos (G0 / G1)
        if upper.startswith("G0") or upper.startswith("G1"):
            tokens = {k.upper(): float(v) for k, v in AXIS_TOKEN_REGEX.findall(upper)}
            is_rel = ("G91" in upper) or (not self._sim_abs_mode and "G90" not in upper)
            with self._sim_lock:
                self._sim_state = "Run"
            self._emit_sim_status()
            time.sleep(random.uniform(0.14, 0.28))
            with self._sim_lock:
                if is_rel:
                    self._sim_x += tokens.get("X", 0.0)
                    self._sim_y += tokens.get("Y", 0.0)
                    self._sim_z += tokens.get("Z", 0.0)
                else:
                    if "X" in tokens:
                        self._sim_x = tokens["X"]
                    if "Y" in tokens:
                        self._sim_y = tokens["Y"]
                    if "Z" in tokens:
                        self._sim_z = tokens["Z"]
                self._sim_state = "Idle"
            self._emit_sim_status()
            return True, "ok"

        # Cualquier otro G-code genérico válido
        time.sleep(random.uniform(0.10, 0.18))
        return True, "ok"

    # ------------------------------------------------------------------
    # Hilos seriales reales (Arduino Uno + CNC Shield V3)
    # ------------------------------------------------------------------
    def _rx_loop(self) -> None:
        """Lee continuamente líneas de GRBL sin bloquear la interfaz."""
        while not self._stop_event.is_set() and self._connected:
            ser = self._serial
            if ser is None or not ser.is_open:
                break
            try:
                raw_bytes = ser.readline()
            except (serial.SerialException, OSError) as exc:
                if self._connected:
                    self._emit_log("ERR", f"Se perdió la conexión serial: {exc}")
                    threading.Thread(target=self.disconnect, daemon=True).start()
                break

            if not raw_bytes:
                continue

            line = raw_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            if line.startswith("<") and line.endswith(">"):
                self._parse_status_line(line)
                continue

            self._emit_log("RX", line)

            # Captura líneas de EEPROM ($100=80.000, $101=80.000, $102=400.000, etc.)
            setting_match = GRBL_PARAM_REGEX.match(line)
            if setting_match:
                self._record_grbl_setting(setting_match.group(1), float(setting_match.group(2)))

            lower = line.lower()
            if lower == "ok" or lower.startswith("error:") or lower.startswith("alarm:"):
                with self._pending_cmd_lock:
                    if self._active_cmd is not None:
                        self._active_cmd.success = lower == "ok"
                        self._active_cmd.response = line
                        self._active_cmd.done_event.set()
                        self._active_cmd = None

    def _tx_loop(self) -> None:
        """Despacha comandos G-code de la cola esperando el `ok` de GRBL entre líneas."""
        while not self._stop_event.is_set() and self._connected:
            try:
                cmd = self._cmd_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            if not self._connected or self._serial is None:
                cmd.success = False
                cmd.response = "No conectado"
                cmd.done_event.set()
                continue

            with self._pending_cmd_lock:
                self._active_cmd = cmd

            payload = (cmd.gcode.strip() + "\n").encode("ascii", errors="ignore")
            try:
                with self._serial_lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(payload)
                        self._serial.flush()
                self._emit_log("TX", cmd.gcode.strip())
            except (serial.SerialException, OSError) as exc:
                with self._pending_cmd_lock:
                    self._active_cmd = None
                cmd.success = False
                cmd.response = str(exc)
                cmd.done_event.set()
                self._emit_log("ERR", f"Error enviando '{cmd.gcode}': {exc}")
                continue

            finished = cmd.done_event.wait(timeout=cmd.timeout_s)
            if not finished:
                with self._pending_cmd_lock:
                    if self._active_cmd is cmd:
                        self._active_cmd = None
                cmd.success = False
                cmd.response = "Tiempo de espera agotado (timeout)"
                self._emit_log("ERR", f"Timeout esperando respuesta para: {cmd.gcode}")

    def _poll_loop(self) -> None:
        """Consulta periódicamente `?` para actualizar posición y estado en la UI."""
        while not self._stop_event.is_set() and self._connected and not self._simulation_mode:
            interval = float(
                self._get_config().get("serial", {}).get("status_poll_interval_s", 0.25)
            )
            time.sleep(max(0.1, interval))
            if self._connected and not self._stop_event.is_set():
                self.query_status()

    def _parse_status_line(self, line: str) -> None:
        match = STATUS_REGEX.match(line)
        if not match:
            return
        state = match.group("state")
        fields = match.group("fields") or ""
        x, y, z = self._status.x, self._status.y, self._status.z
        pos_match = POS_REGEX.search(fields)
        if pos_match:
            x = float(pos_match.group(1))
            y = float(pos_match.group(2))
            z = float(pos_match.group(3))
        self._emit_status(MachineStatus(state=state, x=x, y=y, z=z, raw=line, simulated=False))

    # ------------------------------------------------------------------
    # Comandos en tiempo real y cola G-code
    # ------------------------------------------------------------------
    def _send_realtime_byte(self, byte_data: bytes, log_label: Optional[str] = None) -> bool:
        if not self._connected or self._serial is None:
            return False
        try:
            with self._serial_lock:
                if self._serial and self._serial.is_open:
                    self._serial.write(byte_data)
                    self._serial.flush()
            if log_label:
                self._emit_log("TX", log_label)
            return True
        except Exception as exc:
            self._emit_log("ERR", f"Error en comando en tiempo real: {exc}")
            return False

    def query_status(self) -> bool:
        if self._simulation_mode:
            self._emit_sim_status()
            self._emit_log("SIM", f"<< {self._status.raw}")
            return True
        return self._send_realtime_byte(b"?", log_label=None)

    def _record_grbl_setting(self, key: str, value: float) -> None:
        with self._grbl_settings_lock:
            self._grbl_settings[key] = value
            snapshot = dict(self._grbl_settings)
        if self._on_grbl_settings:
            try:
                self._on_grbl_settings(snapshot)
            except Exception:
                pass

    def get_cached_grbl_settings(self) -> Dict[str, float]:
        with self._grbl_settings_lock:
            return dict(self._grbl_settings)

    def read_grbl_settings(self) -> None:
        """Solicita a GRBL la lectura completa de parámetros EEPROM (`$$`)."""
        self.enqueue_gcode("$$")

    def write_steps_per_mm(
        self, x_steps: float, y_steps: float, z_steps: float
    ) -> Tuple[bool, str]:
        """Escribe los parámetros de calibración pasos/mm ($100, $101, $102) en EEPROM GRBL."""
        if not self._connected:
            return False, "Conectá el Arduino o activá Modo Simulación antes de escribir en EEPROM."
        prefix = "SIM" if self._simulation_mode else "SYS"
        self._emit_log(
            prefix,
            f"Escribiendo pasos/mm en EEPROM: $100={x_steps:.3f}, $101={y_steps:.3f}, $102={z_steps:.3f}",
        )
        self.enqueue_gcode(f"$100={x_steps:.3f}")
        self.enqueue_gcode(f"$101={y_steps:.3f}")
        self.enqueue_gcode(f"$102={z_steps:.3f}")
        return True, "Pasos/mm ($100, $101, $102) enviados a EEPROM GRBL."

    def cancel_jog(self) -> None:
        """Envía inmediatamente el byte en tiempo real GRBL Jog Cancel (0x85) y vacía cola pendiente."""
        self._clear_queue()
        if not self._connected:
            return
        if self._simulation_mode:
            with self._sim_lock:
                if self._sim_state == "Jog":
                    self._sim_state = "Idle"
            self._emit_log("SIM", ">> 0x85 [CANCELAR JOG]")
            self._emit_sim_status()
            return
        self._send_realtime_byte(b"\x85", log_label="0x85 [CANCELAR JOG]")

    def emergency_stop(self) -> None:
        """Parada de emergencia inmediata: envía `!` (Feed Hold) y vacía la cola pendiente."""
        self._clear_queue()
        if self._simulation_mode:
            with self._sim_lock:
                self._sim_state = "Hold"
            self._emit_log("SIM", ">> ! [PARADA DE EMERGENCIA - FEED HOLD]")
            self._emit_sim_status()
            return
        self._send_realtime_byte(b"!", log_label="! [PARADA DE EMERGENCIA - FEED HOLD]")
        self._emit_log("SYS", "Parada de emergencia activada (!). Usá 'Reiniciar' o 'Desbloquear ($X)' para continuar.")

    def resume(self) -> None:
        """Reanuda el movimiento tras una pausa (`~`)."""
        if self._simulation_mode:
            with self._sim_lock:
                self._sim_state = "Idle"
            self._emit_log("SIM", ">> ~ [REANUDAR]")
            self._emit_sim_status()
            return
        self._send_realtime_byte(b"~", log_label="~ [REANUDAR]")

    def soft_reset(self) -> None:
        """Envía Ctrl-X (0x18) para reiniciar GRBL y cancela comandos pendientes."""
        self._clear_queue()
        if self._simulation_mode:
            with self._sim_lock:
                self._sim_state = "Idle"
                self._sim_abs_mode = True
            self._emit_log("SIM", ">> Ctrl-X [REINICIO SUAVE GRBL]")
            self._emit_log("SIM", "<< Grbl 1.1h [SIMULACIÓN]")
            self._emit_sim_status()
            return
        with self._pending_cmd_lock:
            if self._active_cmd is not None:
                self._active_cmd.success = False
                self._active_cmd.response = "Reset GRBL"
                self._active_cmd.done_event.set()
                self._active_cmd = None
        self._send_realtime_byte(b"\x18", log_label="Ctrl-X [REINICIO SUAVE GRBL]")

    def enqueue_gcode(
        self,
        gcode: str,
        timeout_s: Optional[float] = None,
        wait: bool = False,
    ) -> Tuple[bool, str]:
        """Encola una línea G-code. Por defecto no bloquea el hilo llamador."""
        if not self._connected:
            msg = "No hay conexión con el Arduino (activá Modo Simulación o conectá un puerto COM)."
            self._emit_log("ERR", msg)
            return False, msg

        clean = gcode.strip()
        if not clean:
            return False, "Comando vacío."

        if clean == "!":
            self.emergency_stop()
            return True, "ok"
        if clean == "~":
            self.resume()
            return True, "ok"
        if clean == "?":
            self.query_status()
            return True, "ok"

        default_timeout = float(
            self._get_config().get("serial", {}).get("command_timeout_s", 30.0)
        )
        cmd = _QueuedCommand(
            gcode=clean,
            timeout_s=timeout_s if timeout_s is not None else default_timeout,
            done_event=threading.Event(),
        )
        self._cmd_queue.put(cmd)

        if wait:
            cmd.done_event.wait(timeout=cmd.timeout_s + 1.0)
            return cmd.success, cmd.response
        return True, "Encolado"

    # ------------------------------------------------------------------
    # Movimientos de alto nivel (Fluidez simultánea X+Y+Z)
    # ------------------------------------------------------------------
    def home(self) -> None:
        """Ejecuta el ciclo de homing físico o simulado (`$H`)."""
        prefix = "SIM" if self._simulation_mode else "SYS"
        self._emit_log(prefix, "Ejecutando Homing completo ($H)...")
        self.enqueue_gcode("$H", timeout_s=60.0)
        self.enqueue_gcode("G21")
        self.enqueue_gcode("G90")

    def unlock(self) -> None:
        """Desbloquea el estado ALARM de GRBL (`$X`)."""
        self.enqueue_gcode("$X")

    def move_simultaneous(
        self,
        x: float,
        y: float,
        z: float,
        rapid: bool = True,
        feed_rate: Optional[float] = None,
    ) -> None:
        """Mueve los 3 ejes X, Y y Z de forma simultánea en UNA sola línea G-code."""
        if rapid:
            line = f"G0 X{x:.3f} Y{y:.3f} Z{z:.3f}"
        else:
            f = feed_rate if feed_rate is not None else float(
                self._get_config().get("motion", {}).get("default_jog_feed_mm_min", 1800.0)
            )
            line = f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{f:.1f}"
        self.enqueue_gcode(line)

    def jog_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        feed_rate: Optional[float] = None,
    ) -> None:
        """Movimiento manual incremental de eje único (o ejes activos) vía GRBL 1.1 `$J=`.

        Solo incluye los ejes distintos de cero (ej: `$J=G91 G21 X-10.0 F1200`) para evitar
        enviar `Y0.000 Z0.000` innecesarios al mover un solo eje.
        """
        axis_parts: List[str] = []
        if abs(dx) > 1e-6:
            axis_parts.append(f"X{_fmt_gcode_num(dx)}")
        if abs(dy) > 1e-6:
            axis_parts.append(f"Y{_fmt_gcode_num(dy)}")
        if abs(dz) > 1e-6:
            axis_parts.append(f"Z{_fmt_gcode_num(dz)}")
        if not axis_parts:
            return

        f = feed_rate if feed_rate is not None else float(
            self._get_config().get("motion", {}).get("default_jog_feed_mm_min", 1200.0)
        )
        feed_str = f"{f:.0f}" if abs(f - round(f)) < 1e-3 else f"{f:.1f}"
        line = f"$J=G91 G21 {' '.join(axis_parts)} F{feed_str}"
        self.enqueue_gcode(line)

    def dwell(self, seconds: float) -> None:
        """Pausa sincronizada en GRBL (`G4 P<seg>`) para esperar la acción física del servo."""
        if seconds > 0:
            self.enqueue_gcode(f"G4 P{seconds:.2f}")

    # ------------------------------------------------------------------
    # Control de garra SG90 (Pin D11 / grbl-servo)
    # ------------------------------------------------------------------
    def set_gripper_pwm(self, pwm_value: int, dwell: bool = True) -> None:
        """Ajusta el ángulo del servo SG90 mediante `M3 S<valor>`."""
        cfg = self._get_config().get("gripper", {})
        min_pwm = int(cfg.get("min_pwm", 0))
        max_pwm = int(cfg.get("max_pwm", 255))
        clamped = max(min_pwm, min(max_pwm, int(pwm_value)))
        self.enqueue_gcode(f"M3 S{clamped}")
        if dwell:
            self.dwell(float(cfg.get("dwell_seconds", 0.45)))

    def open_gripper(self) -> None:
        cfg = self._get_config().get("gripper", {})
        self.set_gripper_pwm(int(cfg.get("open_pwm", 65)), dwell=True)
        if cfg.get("disable_pwm_after_move", False):
            self.disable_gripper_pwm()

    def close_gripper(self) -> None:
        cfg = self._get_config().get("gripper", {})
        self.set_gripper_pwm(int(cfg.get("close_pwm", 155)), dwell=True)

    def disable_gripper_pwm(self) -> None:
        """Apaga la señal PWM del servo (`M5`) para evitar zumbido o calentamiento."""
        self.enqueue_gcode("M5")

    def test_gripper_cycle(self) -> None:
        """Ciclo rápido de prueba física o simulada: abrir -> cerrar -> abrir."""
        prefix = "SIM" if self._simulation_mode else "SYS"
        self._emit_log(prefix, "Iniciando ciclo de prueba de garra (Abrir -> Cerrar -> Abrir)...")
        self.open_gripper()
        self.close_gripper()
        self.open_gripper()

    # ------------------------------------------------------------------
    # Rutinas compuestas: Standby, Reloj de Ajedrez y Pick & Place
    # ------------------------------------------------------------------
    def go_to_standby(self) -> None:
        """Desplaza el brazo en movimiento simultáneo X+Y+Z a la posición neutral de espera."""
        motion = self._get_config().get("motion", {})
        standby = motion.get("standby_position", {"x": 10.0, "y": 10.0, "z": 45.0})
        z_safe = float(motion.get("z_safe_mm", 45.0))
        self.move_simultaneous(
            float(standby.get("x", 10.0)),
            float(standby.get("y", 10.0)),
            float(standby.get("z", z_safe)),
            rapid=True,
        )

    def go_to_safe_z(self) -> None:
        """Eleva el eje Z manteniendo X e Y actuales en una sola instrucción G0."""
        z_safe = float(self._get_config().get("motion", {}).get("z_safe_mm", 45.0))
        self.move_simultaneous(self._status.x, self._status.y, z_safe, rapid=True)

    def press_chess_clock(self, return_standby: bool = True) -> None:
        """Viaja al reloj de ajedrez, presiona el botón, retrae Z y vuelve a espera."""
        cfg = self._get_config()
        motion = cfg.get("motion", {})
        clock = cfg.get("chess_clock", {})

        z_safe = float(motion.get("z_safe_mm", 45.0))
        plunge_feed = float(motion.get("plunge_feed_mm_min", 900.0))
        cx = float(clock.get("x", 280.0))
        cy = float(clock.get("y", 140.0))
        cz = float(clock.get("z_press", 12.0))
        press_dwell = float(clock.get("press_dwell_seconds", 0.35))

        prefix = "SIM" if self._simulation_mode else "SYS"
        self._emit_log(
            prefix,
            f"Presionando reloj de ajedrez en X={cx:.1f} Y={cy:.1f} Z={cz:.1f}...",
        )
        self.move_simultaneous(cx, cy, z_safe, rapid=True)
        self.move_simultaneous(cx, cy, cz, rapid=False, feed_rate=plunge_feed)
        self.dwell(press_dwell)
        self.move_simultaneous(cx, cy, z_safe, rapid=True)
        if return_standby:
            self.go_to_standby()

    def execute_pick_and_place(
        self,
        src_xy: Tuple[float, float],
        dst_xy: Tuple[float, float],
        is_capture: bool = False,
        press_clock_after: Optional[bool] = None,
    ) -> None:
        """Ejecuta una jugada física/simulada completa con trayectoria suave y opcional captura + reloj."""
        cfg = self._get_config()
        motion = cfg.get("motion", {})
        clock_cfg = cfg.get("chess_clock", {})

        z_safe = float(motion.get("z_safe_mm", 45.0))
        z_pick = float(motion.get("z_pick_mm", 5.0))
        z_place = float(motion.get("z_place_mm", 7.0))
        plunge_feed = float(motion.get("plunge_feed_mm_min", 900.0))

        if is_capture:
            cap = motion.get("capture_zone", {})
            cap_x = float(cap.get("x", 290.0))
            cap_y = float(cap.get("y", 20.0)) + (self._capture_count % 8) * float(cap.get("step_y", 28.0))
            cap_z = float(cap.get("z_drop", z_place))
            self._capture_count += 1
            prefix = "SIM" if self._simulation_mode else "SYS"
            self._emit_log(prefix, f"Retirando pieza capturada hacia zona de descarte ({cap_x:.1f}, {cap_y:.1f})...")
            self._single_transfer(dst_xy, (cap_x, cap_y), z_safe, z_pick, cap_z, plunge_feed)

        self._single_transfer(src_xy, dst_xy, z_safe, z_pick, z_place, plunge_feed)

        should_press_clock = (
            bool(clock_cfg.get("enabled", True))
            if press_clock_after is None
            else press_clock_after
        )
        if should_press_clock:
            self.press_chess_clock(return_standby=True)
        else:
            self.go_to_standby()

    def _single_transfer(
        self,
        src_xy: Tuple[float, float],
        dst_xy: Tuple[float, float],
        z_safe: float,
        z_pick: float,
        z_place: float,
        plunge_feed: float,
    ) -> None:
        sx, sy = src_xy
        dx, dy = dst_xy

        self.open_gripper()
        self.move_simultaneous(sx, sy, z_safe, rapid=True)
        self.move_simultaneous(sx, sy, z_pick, rapid=False, feed_rate=plunge_feed)
        self.close_gripper()
        self.move_simultaneous(sx, sy, z_safe, rapid=True)
        self.move_simultaneous(dx, dy, z_safe, rapid=True)
        self.move_simultaneous(dx, dy, z_place, rapid=False, feed_rate=plunge_feed)
        self.open_gripper()
        self.move_simultaneous(dx, dy, z_safe, rapid=True)

    # ------------------------------------------------------------------
    # Perfiles de velocidad y aceleración GRBL ($110-$112, $120-$122)
    # ------------------------------------------------------------------
    def apply_grbl_preset(self, preset_key: str) -> Tuple[bool, str]:
        """Envía a GRBL (real o simulado) los parámetros de velocidad y aceleración del perfil."""
        presets = self._get_config().get("grbl_presets", {}).get("presets", {})
        preset = presets.get(preset_key)
        if not preset:
            return False, f"Perfil GRBL desconocido: {preset_key}"

        if not self._connected:
            return False, "Conectá el Arduino o activá Modo Simulación antes de aplicar un perfil GRBL."

        prefix = "SIM" if self._simulation_mode else "SYS"
        self._emit_log(prefix, f"Aplicando perfil GRBL '{preset.get('label', preset_key)}'...")
        for param in ("$110", "$111", "$112", "$120", "$121", "$122"):
            if param in preset:
                val = float(preset[param])
                self.enqueue_gcode(f"{param}={val:.1f}")
        return True, f"Perfil '{preset.get('label', preset_key)}' enviado a GRBL."
