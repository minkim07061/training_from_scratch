"""Make the round3 ``training_lab`` package importable for its tests.

This inserts the round3 folder onto ``sys.path`` so tests can simply do
``from training_lab.optim import AdamW`` regardless of where pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROUND3_DIR = Path(__file__).resolve().parent
if str(_ROUND3_DIR) not in sys.path:
    sys.path.insert(0, str(_ROUND3_DIR))
