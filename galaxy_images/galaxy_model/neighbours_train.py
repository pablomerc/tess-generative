"""
Backward-compatible wrapper around the unified training entrypoint.

You can still run:
  python galaxy_images/galaxy_model/neighbours_train.py

Internally this delegates to:
  python -m galaxy_images.galaxy_model.train --config galaxy_images/galaxy_model/configs/neighbors_default.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from galaxy_images.galaxy_model.train import main as unified_train_main


def main() -> None:
    default_config = Path(__file__).resolve().parent / "configs" / "neighbors_default.json"
    argv = ["--config", str(default_config), *sys.argv[1:]]
    unified_train_main(argv)


if __name__ == "__main__":
    main()
