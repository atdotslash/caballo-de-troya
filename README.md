# Caballo de Troya — Controlador de Brazo Robótico SCARA para Ajedrez

Aplicación de escritorio modular en Python para **Windows 10** diseñada para controlar un brazo robótico de 3 ejes (Arduino Uno R3 + CNC Shield V3 + firmware `grbl-servo`) con cámara cenital (ArUco `DICT_5X5_50`), garra microservo SG90, reloj de ajedrez automático y anuncios por voz offline en Español de Argentina (`es-AR`).

## Estructura del Proyecto

```text
caballo-de-troya/
├── main.py                      # Punto de entrada principal
├── config.json                  # Configuración persistente (GRBL, Z, garra, reloj, calibración)
├── requirements.txt             # Dependencias Python
├── assets/
│   ├── frases_humor.json        # Frases sarcásticas en Español (Argentina) por categoría
│   ├── sounds/                  # Clips .mp3/.wav por categoría (inicio, demora, captura, jaque, victoria)
│   └── aruco_32_piezas_DICT_5X5_50.png
├── src/
│   ├── __init__.py
│   ├── config_manager.py        # Lectura/escritura atómica de config.json
│   ├── vision.py                # Cámara cenital, ArUco DICT_5X5_50 y homografía 4 esquinas
│   ├── robot_controller.py      # Serial GRBL no bloqueante + Modo Simulación, G0 simultáneo X/Y/Z, garra y reloj
│   ├── audio_manager.py         # TTS offline (pyttsx3 es-AR) + pygame.mixer (.mp3/.wav) + temporizador de demora
│   ├── chess_engine.py          # Lógica python-chess, detección de jugadas y soporte UCI
│   └── gui.py                   # Interfaz moderna en CustomTkinter (100% Español Argentina)
└── tools/
    └── generate_aruco_sheet.py  # Generador de planilla imprimible con los 32 marcadores ArUco
```

## Notas de Hardware (`Arduino Uno + CNC Shield V3 + grbl-servo`)

- **Comunicación:** USB a `115200` baudios.
- **Finales de carrera (X, Y, Z):** Soportan `$H` (Homing). Recordá que en GRBL 1.1 con PWM variable (`grbl-servo`), el pin **D11** (serigrafiado como `Z+` en CNC Shield V3) corresponde a la señal PWM del servo **SG90**, mientras que el límite físico de **Z** suele reubicarse en el pin **D12** (`SpnEn`).
- **Garra SG90:** Se comanda mediante `M3 S<valor>` (por defecto `0–255` en `grbl-servo`) y `M5` para desactivar el PWM.

## Instalación (Windows 10 con entorno virtual `.venv`)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Ejecución

```powershell
.venv\Scripts\python.exe main.py
```

### Generar marcadores ArUco (IDs 0–31)

```powershell
.venv\Scripts\python.exe tools\generate_aruco_sheet.py
```

## Flujo de Validación de Hardware (Fase 1)

1. Conectá el Arduino Uno por USB y hacé clic en **Actualizar** → **Conectar**.
2. Ejecutá **Hacer Homing ($H)** (tené a mano el botón rojo **Parada de Emergencia (!)**).
3. En la pestaña **Calibración y Pruebas**:
   - Probá el movimiento incremental con el **Jog X / Y / Z**.
   - Validá el movimiento fluido simultáneo en 3 ejes con **Movimiento Simultáneo (G0 X Y Z)**.
   - Ajustá los ángulos de apertura y cierre de la garra SG90 con el slider (`M3 S<val>`), guardalos y ejecutá **Probar Garra**.
   - Posicioná el brazo sobre el botón del reloj de ajedrez, hacé clic en **Guardar Pos. Actual como Reloj** y validá con **Presionar Reloj**.
   - Probá la voz en Español de Argentina con **Probar Voz**.
4. Iniciá la cámara cenital y hacé clic en **Calibrar Tablero** marcando en orden las 4 esquinas exteriores: **A8 → H8 → H1 → A1**.
