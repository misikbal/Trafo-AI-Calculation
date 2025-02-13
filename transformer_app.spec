# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['transformer_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('ag_yassi_tel.joblib', '.'),
        ('YG_Emaye.joblib', '.'),
        ('folyo.joblib', '.'),
        ('logo.jpeg', '.'),
        ('datsanstlogo.ico', '.')
    ],
    hiddenimports=[
        'sklearn.ensemble._forest',
        'sklearn.tree._tree',
        'sklearn.utils._typedefs',
        'sklearn.metrics._pairwise_distances_reduction._datasets_pair',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'pandas',
        'numpy',
        'joblib',
        'openpyxl'
    ],
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
    name='Transformer Prediction System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='datsanstlogo.ico',
    version='file_version_info.txt'
) 