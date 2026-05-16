"""
Same Modal NVFP4 export as repo-root ``v3.py``, runnable from this package:

  cd /path/to/Italk && modal run fast-flashtalk-main-nunchaku/scripts/quantize_nvfp4_w4a4_modal.py

See ``v3.py`` for full documentation.
"""
from __future__ import annotations

import runpy
from pathlib import Path

_ITALK_ROOT = Path(__file__).resolve().parents[2]
_V3 = _ITALK_ROOT / "v3.py"

if not _V3.is_file():
    raise FileNotFoundError(f"Expected {_V3}")

runpy.run_path(str(_V3), run_name="__main__")
