"""Model registry: placeholder until M4 (returns no predictor when no model is trained)."""

from __future__ import annotations

from typing import Optional, Tuple


def load_predictor(name: Optional[str] = None) -> Tuple[Optional[object], Optional[int]]:
    return None, None
