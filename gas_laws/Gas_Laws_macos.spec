# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


target_arch = os.environ.get("PYINSTALLER_TARGET_ARCH") or None

datas = (
    collect_data_files("bokeh")
    + collect_data_files("tornado")
    + [("Gas_Laws.py", "."), ("data", "data")]
)

hiddenimports = (
    collect_submodules("bokeh")
    + collect_submodules("tornado")
    + ["jinja2", "numpy", "pandas", "xyzservices"]
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
    name="CHE031_Gas_Laws",
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
    name="CHE031_Gas_Laws",
)

app = BUNDLE(
    collection,
    name="CHE031 Gas Laws.app",
    bundle_identifier="edu.lehigh.srr.che031-gas-laws",
    info_plist={
        "CFBundleDisplayName": "CHE031 Gas Laws",
        "CFBundleName": "CHE031 Gas Laws",
        "NSHighResolutionCapable": True,
    },
    target_arch=target_arch,
    codesign_identity=None,
)
