@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Compilando CaballoDeTroya.exe (Windows 10 - PyInstaller)
echo ============================================================

if not exist ".venv\Scripts\pyinstaller.exe" (
    echo [ERROR] No se encontro PyInstaller en .venv. Instalalo con:
    echo         .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

".venv\Scripts\pyinstaller.exe" --noconfirm --clean build_exe.spec
if errorlevel 1 (
    echo [ERROR] Fallo la compilacion de PyInstaller.
    exit /b 1
)

echo.
echo Copiando configuracion y recursos editables junto al .exe...
copy /Y "config.json" "dist\config.json" >nul
if not exist "dist\assets" mkdir "dist\assets"
xcopy /E /I /Y "assets" "dist\assets" >nul

echo.
echo ============================================================
echo   Listo! Ejecutable generado en: dist\CaballoDeTroya.exe
echo   Podes editar dist\config.json y agregar .mp3 en
echo   dist\assets\sounds\ sin volver a compilar.
echo ============================================================
endlocal
