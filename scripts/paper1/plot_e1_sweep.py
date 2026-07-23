#!/usr/bin/env python3
"""Compatibility entry point for regenerating E1 sweep plots."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper1.run_e1_sweep import main

if __name__ == "__main__":
    raise SystemExit(main(["--plot-only", *sys.argv[1:]]))
