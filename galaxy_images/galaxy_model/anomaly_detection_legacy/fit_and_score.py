"""
Thin wrapper around anomaly_detection/fit_and_score.py that redirects OUTPUT_DIR
to anomaly_detection_legacy/outputs/, so Legacy-side scores never collide with
the HSC-side ones. The detector logic is identical and stays in one place.
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_hsc_dir = _here.parent / "anomaly_detection"
if str(_hsc_dir) not in sys.path:
    sys.path.insert(0, str(_hsc_dir))

import fit_and_score as _orig

_orig.OUTPUT_DIR = _here / "outputs"

if __name__ == "__main__":
    _orig.main()
