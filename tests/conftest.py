"""
Test configuration for pytest.

Ensures the project's `src` directory is on `sys.path` so tests can import
project modules without needing editable installs or environment tweaks.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Prepend project root so `import src` resolves inside tests.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
