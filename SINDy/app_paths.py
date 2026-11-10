"""Filesystem locations owned by the SINDy module.

Resolve packaged assets from this file's location so the application behaves
the same whether Bokeh is started from the repository root or from SINDy/.
"""

from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
