"""Pytest setup: repo root on path and stdlib `code` shadowing fix."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]


@pytest.fixture(autouse=True)
def reset_roi_profile(monkeypatch):
    """Default tests to legacy full-brain profile unless a fixture overrides."""
    monkeypatch.delenv("ILCLDM_ROI_VOLUME_SHAPE", raising=False)
    monkeypatch.delenv("ILCLDM_SKIP_CROP", raising=False)
    import config
    from code.volume_shapes import clear_shape_cache

    config.ROI_VOLUME_SHAPE = None
    config.SKIP_CROP = False
    clear_shape_cache()
    yield
    clear_shape_cache()
