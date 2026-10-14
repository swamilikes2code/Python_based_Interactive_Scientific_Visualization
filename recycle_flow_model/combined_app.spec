# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect static assets and data files for Bokeh, Tornado, and PIL
bokeh_datas = collect_data_files('bokeh')
tornado_datas = collect_data_files('tornado')
pil_datas = collect_data_files('PIL')

# Bundle combined_app.py and image assets into sys._MEIPASS root
app_datas = [
    ('combined_app.py', '.'),
    ('no_recycle.png', '.'),
    ('recycle.png', '.'),
]

all_datas = bokeh_datas + tornado_datas + pil_datas + app_datas

# Collect submodules to prevent runtime missing import exceptions
hidden_modules = (
    collect_submodules('bokeh') +
    collect_submodules('tornado') +
    collect_submodules('PIL') +
    [
        'numpy',
        'jinja2',
        'xyzservices',
    ]
)

block_cipher = None

a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=all_datas,
    hiddenimports=hidden_modules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CHE031_Crystallization_App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False once you wish to hide the CMD window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)