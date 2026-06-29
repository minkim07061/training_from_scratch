"""Pytest import routing for round-two practice.

By default, tests import the completed implementation from ``src/`` via the
project's pyproject configuration. To run the same tests against the no-solution
round-two scaffold, set:

    SCRATCH_TRANSFORMER_TARGET=round2

Example:

    SCRATCH_TRANSFORMER_TARGET=round2 python3 -m pytest tests/test_norm.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


TARGET = os.environ.get("SCRATCH_TRANSFORMER_TARGET", "main").lower()

if TARGET in {"round2", "round2_no_llm"}:
    repo_root = Path(__file__).resolve().parents[1]
    round2_path = repo_root / "round2_no_llm"
    sys.path.insert(0, str(round2_path))

    # If any helper imported the package before this conftest loaded, make sure
    # subsequent imports resolve from round2_no_llm instead of src.
    for module_name in list(sys.modules):
        if module_name == "scratch_transformer" or module_name.startswith("scratch_transformer."):
            del sys.modules[module_name]
elif TARGET not in {"main", "src"}:
    raise RuntimeError(
        "Unknown SCRATCH_TRANSFORMER_TARGET. Use 'main' or 'round2'."
    )


def pytest_report_header() -> str:
    if TARGET in {"round2", "round2_no_llm"}:
        return "scratch_transformer target: round2_no_llm"
    return "scratch_transformer target: src"
