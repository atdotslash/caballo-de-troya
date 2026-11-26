# -*- mode: python ; coding: utf-8 -*-
"""Especificación PyInstaller para generar dist/CaballoDeTroya.exe (Windows 10)."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ctk_datas = collect_data_files("customtkinter")
hiddenimports = (
    collect_submodules("pyttsx3.drivers")
    + [
        "PIL._tkinter_finder",
        "serial.tools.list_ports",
        "serial.tools.list_ports_windows",
        "chess",
        "chess.engine",
        "cv2",
        "pygame",
    ]
)

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=ctk_datas + [("config.json", ".")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="CaballoDeTroya",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
