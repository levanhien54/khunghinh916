"""Cho phép `python -m khunghinh`."""
from __future__ import annotations

from khunghinh.app import main

if __name__ == "__main__":
    raise SystemExit(main())
