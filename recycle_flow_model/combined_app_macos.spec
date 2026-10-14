# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


target_arch = os.environ.get("PYINSTALLER_TARGET_ARCH") or None

datas = (
    collect_data_files("bokeh")
    + collect_data_files("tornado")
    + collect_data_files("PIL")
    + [
        ("combined_app.py", "."),
        ("no_recycle.png", "."),
        ("recycle.png", "."),
    ]
)

hiddenimports = (
    collect_submodules("bokeh")
    + collect_submodules("tornado")
    + collect_submodules("PIL")
    + ["jinja2", "numpy", "xyzservices"]
)

a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
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
    [],
    exclude_binaries=True,
    name="CHE031_Crystallization_App",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="CHE031_Crystallization_App",
)

app = BUNDLE(
    collection,
    name="CHE031 Crystallization.app",
    bundle_identifier="edu.lehigh.srr.che031-crystallization",
    info_plist={
        "CFBundleDisplayName": "CHE031 Crystallization",
        "CFBundleName": "CHE031 Crystallization",
        "NSHighResolutionCapable": True,
    },
    target_arch=target_arch,
    codesign_identity=None,
)
