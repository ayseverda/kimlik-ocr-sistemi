# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# collect_all() her paketin test klasörlerini de (scipy.*.tests, numpy.*.tests,
# skimage.*.tests) topluyordu -- binlerce gereksiz modül, build'i çok yavaşlatıp
# paketi gereksiz yere şişiriyordu. filter_submodules ile bunları eliyoruz.
def _testsiz(name):
    parcalar = name.split('.')
    return 'tests' not in parcalar and 'test' not in parcalar

datas = [('referans_kimlik.jpg', '.'), ('referans_eski_tc.jpg', '.'), ('referans_gocmen.jpg', '.'), ('easyocr_models', 'easyocr_models')]
binaries = []
hiddenimports = []
for paket in ('torch', 'easyocr', 'cv2', 'PySide6', 'numpy', 'scipy', 'skimage'):
    tmp_ret = collect_all(paket, filter_submodules=_testsiz)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['desktop.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KimlikOkuyucu',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KimlikOkuyucu',
)
