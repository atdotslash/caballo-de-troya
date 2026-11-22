# Task: Add Simulation Mode + Sarcastic Humor & Sound Effects Module

Update the existing `caballo-de-troya` project to support hardware-free testing and randomized Argentinian Spanish humor/sound effects for high school students.

## 1. Simulation Mode (`robot_controller.py` & `gui.py`)
- Add a **"Modo Simulación"** toggle switch in the top serial connection bar of `gui.py` (enabled without needing a COM port).
- When active, `robot_controller.py` must route all G-code commands (`$H`, `G0`, `G1`, `M3`, `M5`, `$J=`, etc.) to an internal mock serial handler that:
  - Simulates realistic short execution delays (e.g., 100–300 ms).
  - Automatically returns `ok` and updates the simulated `MPos` (X, Y, Z) coordinates in real time.
  - Logs every simulated G-code command and response in the live G-code console with a `[SIM]` prefix.
- Allow testing Homing, Jog, Simultaneous G0, Gripper, and Chess Clock routines seamlessly in Simulation Mode.

## 2. Humor & Sound Effects (`audio_manager.py`, `assets/frases_humor.json`, `requirements.txt`)
- Add `pygame` to `requirements.txt` and use `pygame.mixer` inside `audio_manager.py` to play short `.mp3` / `.wav` audio clips without blocking the UI or TTS thread.
- Create `assets/frases_humor.json` with 5 categories: `inicio`, `demora`, `captura`, `jaque`, `victoria`. Populate it with sarcastic, playful, school-friendly Argentinian Spanish (`es-AR`) phrases (at least 15 for `inicio`/`victoria`, 25 for `demora`/`captura`/`jaque`).
- Create a folder structure `assets/sounds/<category>/` (`inicio`, `demora`, `captura`, `jaque`, `victoria`) so I can drop custom `.mp3`/`.wav` clips (e.g., *"Tres doritos después"*, *"We Will Rock You"*, *"Fatality"*).
- Implement `play_event(category)` in `audio_manager.py`:
  - Randomly picks between playing a custom audio clip from `assets/sounds/<category>/` (if any exist) or speaking a random phrase from `assets/frases_humor.json` via `pyttsx3` (avoiding immediate repeats).
- Add a configurable **inactivity/delay timer** (`delay_warning_seconds`, default `20` in `config.json`). During a human's turn, if the timer expires, automatically trigger `play_event("demora")` and reset the timer.

## 3. UI Updates (`gui.py` — Strictly Spanish Argentina)
- Add a **"Modo Simulación"** switch in the top bar.
- In the **Calibración y Pruebas** tab, add a **"Probar Humor y Sonidos"** section with quick-test buttons for each category: *Inicio*, *Demora*, *Captura*, *Jaque*, and *Victoria*, plus a toggle to enable/disable the humor timer.