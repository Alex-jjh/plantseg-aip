"""Shared CLI bootstrap: puts the repo root on sys.path and returns Paths."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cw2.paths import Paths  # noqa: E402

P = Paths(REPO)
