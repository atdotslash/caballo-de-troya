# Caballo de Troya — Controlador de Brazo Robótico SCARA para Ajedrez (`v1.1.0`)

Aplicación de escritorio modular en Python para **Windows 10** diseñada para controlar un brazo robótico SCARA de 3 ejes (**Arduino Uno R3 + CNC Shield V3 + firmware `grbl-servo`**) con cámara cenital (ArUco `DICT_5X5_50`), garra microservo SG90, reloj de ajedrez automático, tablero interactivo con IA integrada, **Modo Simulación** sin hardware y motor de humor con clips de audio `.mp3`.

---

## Características Principales

- **Modo Simulación (`Modo Simulación`):** Interruptor en la barra superior para probar todas las rutinas (`$H`, Jog de eje único, `G0` simultáneo, garra `M3`/`M5`, reloj de ajedrez, lectura/escritura EEPROM y partidas completas) sin conectar un Arduino ni puerto COM, con actualización en tiempo real de `MPos (X, Y, Z)` y registros `[SIM]` en la consola G-code.
- **Tablero Interactivo 8×8 e IA Integrada:** Tablero clicable con piezas Unicode (`♔♕♖♗♘♙♚♛♜♝♞♟`), resaltado de jugadas legales, captura y jaque, botón **Nueva Partida** y selector **Jugar contra IA** con motor **Minimax Alfa-Beta (profundidad 2)** incorporado como respaldo automático cuando Stockfish UCI no está configurado.
- **Humor Sarcástico y Efectos de Sonido (`assets/sounds/`):** Reproducción no bloqueante de clips `.mp3`/`.wav` vía `pygame.mixer` en 5 categorías (`inicio`, `demora`, `captura`, `jaque`, `victoria`), modo solo clips MP3 por defecto (`"use_tts": false` en `config.json`), prevención de repeticiones inmediatas y temporizador configurable de demora durante el turno humano (`delay_warning_seconds`).
- **Control GRBL Preciso y Seguro:**
  - Jog incremental de **eje único** (`$J=G91 G21 X-10.0 F1200`) con antirrebote de clic.
  - Botón **Detener Jog (0x85 / Esc)** y liberación de teclas direccionales enviando el byte en tiempo real `\x85` (*Jog Cancel*).
  - **Calibración de Pasos/mm (`$100`, `$101`, `$102`)** directa contra la EEPROM de GRBL desde la pestaña *Configuración* (*Leer de GRBL `$$`* y *Escribir a EEPROM*).
  - Movimiento simultáneo en 3 ejes (`G0 X.. Y.. Z..`), control de garra SG90 en pin `D11` (`M3 S<val>` / `M5`) y secuencia automática de reloj de ajedrez + retorno a espera.
- **Visión Artificial Cenital (`OpenCV 4.10+ / 5.x`):** Detección en vivo de los 32 marcadores ArUco (`DICT_5X5_50`, piezas intercambiables por tipo/color) y asistente de calibración de 4 esquinas (`A8 → H8 → H1 → A1`) mediante homografía píxeles ↔ tablero 8×8 ↔ milímetros de máquina.
- **Interfaz Optimizada sin Scroll (100% Español Argentina):** Distribución a pantalla completa con cámara amplia a la izquierda y panel derecho de 2 columnas sin desplazamiento vertical en *Calibración y Pruebas*.
- **Ejecutable Independiente (`CaballoDeTroya.exe`):** Empaquetado con PyInstaller resolviendo `config.json` y `assets/sounds/` de forma externa junto al `.exe` para poder editar parámetros o sumar nuevos `.mp3` sin recompilar.

---

## Estructura del Proyecto

```text
caballo-de-troya/
├── main.py                      # Punto de entrada principal (compatible con Python y .exe)
├── config.json                  # Configuración persistente (GRBL, Z, pasos/mm, garra, reloj, audio, calibración)
├── requirements.txt             # Dependencias Python (incluye PyInstaller)
├── build_exe.spec               # Especificación PyInstaller para CaballoDeTroya.exe
├── build_exe.bat                # Script de compilación + copia de config.json y assets/ en dist/
├── assets/
│   ├── frases_humor.json        # Frases sarcásticas en Español (Argentina) por categoría
│   ├── sounds/                  # Clips .mp3/.wav (inicio, demora, captura, jaque, victoria)
│   └── aruco_32_piezas_DICT_5X5_50.png
├── src/
│   ├── __init__.py
│   ├── paths.py                 # Resolución de rutas externas/empaquetadas (.exe / dev)
│   ├── config_manager.py        # Lectura/escritura atómica de config.json
│   ├── vision.py                # Cámara cenital, ArUco DICT_5X5_50 y homografía 4 esquinas
│   ├── robot_controller.py      # Serial GRBL no bloqueante, Modo Simulación, Jog eje único, 0x85 y EEPROM
│   ├── audio_manager.py         # Clips MP3 (pygame.mixer), TTS opcional (pyttsx3) y temporizador de demora
│   ├── chess_engine.py          # Lógica python-chess, IA Minimax integrada y soporte UCI (Stockfish)
│   └── gui.py                   # Interfaz moderna en CustomTkinter (100% Español Argentina)
├── tests/
│   └── test_core.py             # Pruebas unitarias automatizadas
└── tools/
    └── generate_aruco_sheet.py  # Generador de planilla imprimible con los 32 marcadores ArUco
```

---

## Cómo Ejecutar

### Opción A — Ejecutable Independiente (`CaballoDeTroya.exe`)

1. Si todavía no compilaste el ejecutable, corré desde la raíz del proyecto:
   ```powershell
   build_exe.bat
   ```
2. Abrí la carpeta `dist\` y hacé doble clic en **`dist\CaballoDeTroya.exe`**.
3. Podés editar `dist\config.json` o agregar nuevos archivos `.mp3` dentro de `dist\assets\sounds\<categoria>\` sin volver a compilar.

### Opción B — Desde Código Fuente (entorno virtual `.venv`)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

### Generar marcadores ArUco (IDs 0–31) y correr pruebas

```powershell
.venv\Scripts\python.exe tools\generate_aruco_sheet.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## Notas de Hardware (`Arduino Uno + CNC Shield V3 + grbl-servo`)

- **Comunicación:** USB a `115200` baudios.
- **Finales de carrera (X, Y, Z):** Soportan `$H` (Homing). En GRBL 1.1 con PWM variable (`grbl-servo`), el pin **D11** (`Z+` en CNC Shield V3) entrega la señal PWM al servo **SG90**, mientras que el límite físico de **Z** suele conectarse en **D12** (`SpnEn`).
- **Calibración de Pasos/mm (`$100`, `$101`, `$102`):** Si un paso de `10 mm` recorre una distancia distinta a la real, abrí la pestaña **Configuración**, hacé clic en **Leer de GRBL ($$)**, ajustá `$100` (X), `$101` (Y) y `$102` (Z) con regla de tres (`nuevo_valor = valor_actual * distancia_pedida / distancia_medida`) y presioná **Escribir a EEPROM ($100-$102)**.

---

## Historial de Versiones (Changelog)

### `v1.1.0` — Simulación, Tablero Interactivo, Humor/Audio, Fix Jog y Empaquetado `.exe`
- **Modo Simulación:** Controlador GRBL virtual integrado sin puerto COM con retardos realistas (`100–300 ms`), actualización de `MPos` en vivo y prefijo `[SIM]` en consola.
- **Tablero Interactivo 8×8 e IA:** Nuevo tablero clicable Unicode en *Tablero y Partida* con resaltado de movimientos legales, botón *Nueva Partida*, interruptor *Jugar contra IA* e IA interna Minimax Alfa-Beta (profundidad 2) como respaldo de Stockfish.
- **Motor de Humor y Clips MP3:** Integración de `pygame.mixer` para reproducción no bloqueante de los clips `.mp3` en `assets/sounds/` (`inicio`, `demora`, `captura`, `jaque`, `victoria`), modo solo audio por defecto (`"use_tts": false`), prevención de repeticiones inmediatas y temporizador configurable de demora.
- **Corrección de Jog y Calibración EEPROM:** Jog incremental de eje único (`$J=G91 G21 X-10.0 F1200`) con antirrebote, botón *Detener Jog (0x85 / Esc)* y atajos de teclado con *Jog Cancel* (`\x85`), más lectura/escritura directa de pasos/mm (`$100`, `$101`, `$102`) en la pestaña *Configuración*.
- **Rediseño de Interfaz (Sin Scroll):** Panel derecho ampliado en 2 columnas dentro de *Calibración y Pruebas*, eliminación de recortes de texto y máximo aprovechamiento del visor cenital.
- **Empaquetado Windows `.exe`:** Agregados `src/paths.py`, `build_exe.spec` y `build_exe.bat` para generar `dist/CaballoDeTroya.exe` con lectura externa de `config.json` y `assets/sounds/`.

### `v1.0.0` — Base SCARA + GRBL + OpenCV (Fase 1)
- Arquitectura modular inicial (`config_manager.py`, `vision.py`, `robot_controller.py`, `audio_manager.py`, `chess_engine.py`, `gui.py`).
- Comunicación serial no bloqueante con Arduino Uno + CNC Shield V3 (`grbl-servo`), homing `$H`, parada de emergencia `!`, movimiento simultáneo 3 ejes (`G0 X Y Z`), control de garra SG90 (`M3`/`M5`), rutina de reloj de ajedrez y perfiles GRBL (`$110–$122`).
- Detección de 32 marcadores ArUco (`DICT_5X5_50`, OpenCV 4.10+/5.x), calibración de 4 esquinas por homografía y generador de planilla imprimible.
- Interfaz gráfica moderna en CustomTkinter 100% en Español de Argentina y pruebas unitarias base.
