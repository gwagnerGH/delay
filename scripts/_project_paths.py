"""Shared import paths for scripts moved under the scripts directory."""

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent


def configure_paths():
    for path in (SCRIPTS_DIR, PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
