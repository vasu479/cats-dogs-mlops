"""Ensure the repository root is importable regardless of how pytest is invoked.

Keeps `pytest`, `python -m pytest` and the CI runner all resolving `src.*` and
`app.*` identically on Windows and Linux.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
