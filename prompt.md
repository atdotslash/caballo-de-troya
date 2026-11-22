# Role & Objective
Act as a Senior Robotics & Computer Vision Software Engineer. Build a modular, production-ready Python desktop application for **Windows 10** to control a 3-axis SCARA-style robotic arm that plays chess autonomously.

# Hardware & Firmware Context
- **Controller:** Arduino Uno R3 (generic) + CNC Shield V3 connected via USB (Windows COM port, 115200 baud).
- **Firmware:** `grbl-servo` (GRBL 1.1 fork supporting 50Hz PWM for servos).
- **Actuators:**
  - 3x NEMA 17 stepper motors (X, Y, Z axes) with physical end-stops installed on **X, Y, and Z** (supports `$H` homing).
  - 1x SG90 9g microservo gripper connected to pin `D11` (`Z+` on CNC Shield V3), controlled via G-code `M3 S<angle_value>` (close/open) and `M5`.
- **Vision:** Overhead gantry webcam (top-down / zenithal view) pointing straight down at the chessboard.
- **Audio:** Windows 10 PC speakers (no Arduino pins used for audio).

# Core Functional Requirements

## 1. Computer Vision (`vision.py`)
- Use `opencv-contrib-python` (compatible with OpenCV 4.10+ / 5.x `cv2.aruco.ArucoDetector`).
- Detect 32 unique ArUco markers from dictionary `cv2.aruco.DICT_5X5_50`:
  - **White pieces:** IDs `0–15` (Rooks: 0,7; Knights: 1,6; Bishops: 2,5; Queen: 3; King: 4; Pawns: 8–15).
  - **Black pieces:** IDs `16–31` (Rooks: 16,23; Knights: 17,22; Bishops: 18,21; Queen: 19; King: 20; Pawns: 24–31).
  - Pieces of the same type/color must be interchangeable regardless of their specific ID.
- Implement a **4-corner board calibration tool** (perspective transform / homography) to map the 8x8 chessboard squares from camera pixels to real-world machine coordinates (millimeters).

## 2. Smooth Motion & Serial Controller (`robot_controller.py`)
- Use `pyserial` for non-blocking communication with the Arduino COM port (wait for `ok` responses from GRBL).
- **Fluid Simultaneous Motion:** Always combine X, Y, and Z targets in a single G-code line (e.g., `G0 X50.0 Y30.0 Z10.0`) so axes move simultaneously instead of sequentially.
- Include configurable parameters in a `config.json` file for:
  - Safe Z travel height, pick Z height, and place Z height.
  - Gripper open/close `M3 S<val>` values.
  - **Chess Clock Coordinate:** A fixed `(X, Y, Z_press)` coordinate. After completing a move, the arm must automatically travel to the chess clock, lower Z to press the button, retract Z, and return to a neutral/home standby position.
  - GRBL speed (`$110-$112`) and acceleration (`$120-$122`) tuning presets.

## 3. Chess Logic & Voice Feedback (`audio_manager.py` & `chess_engine.py`)
- Integrate `python-chess` to validate board states, detect human moves via ArUco changes, and optionally connect to a local engine (like Stockfish) or manual move input for testing.
- Implement an offline Text-to-Speech (TTS) module (e.g., `pyttsx3` or `edge-tts` + `pygame`) running in a background thread so it never blocks the UI or robot motion.
- Announce game events in **Argentinian Spanish (`es-AR`)**: `"Jaque"`, `"Jaque Mate"`, `"Tablas"`, `"Turno de las blancas"`, `"Movimiento inválido"`, etc.

## 4. User Interface (`gui.py`)
- Build a clean, modern desktop GUI using `customtkinter` (or `PyQt6`).
- **STRICT LANGUAGE RULE:** All UI labels, buttons, tooltips, dialogs, and status messages MUST be in **Spanish (Argentina)** (e.g., *"Conectar"*, *"Hacer Homing ($H)"*, *"Calibrar Tablero"*, *"Probar Garra"*, *"Presionar Reloj"*).
- Required UI panels:
  1. Live webcam feed with detected ArUco bounding boxes, piece names, and 8x8 grid overlay.
  2. Serial connection bar (auto-detect COM ports, Connect/Disconnect, Homing `$H`, Emergency Stop `!` / Reset).
  3. Calibration & Testing tab (manual Jog X/Y/Z, servo open/close slider, save chess clock coordinates, test TTS voice).
  4. Live G-code console log.

# Deliverables & Execution Plan
Do not write everything in a single monolithic file. Start by:
1. Proposing the project folder structure and `requirements.txt`.
2. Creating the configuration schema (`config.json`).
3. Implementing Phase 1: `robot_controller.py`, `audio_manager.py`, and a working `gui.py` with live ArUco detection and manual/simultaneous G-code testing so I can validate the physical hardware first.