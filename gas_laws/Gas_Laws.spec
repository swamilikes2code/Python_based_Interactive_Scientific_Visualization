# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files and submodules for Bokeh and Tornado
bokeh_datas = collect_data_files('bokeh')
tornado_datas = collect_data_files('tornado')

# Bundle Gas_Laws.py and the data folder into sys._MEIPASS root
app_datas = [
    ('Gas_Laws.py', '.'),
    ('data', 'data'),
]

all_datas = bokeh_datas + tornado_datas + app_datas

# Explicitly collect submodules to prevent missing import errors at runtime
hidden_modules = (
    collect_submodules('bokeh') +
    collect_submodules('tornado') +
    [
        'pandas',
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
    name='Gas_Laws',
    debug=True,  # Enables PyInstaller bootloader debug statements
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep command prompt window open
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)