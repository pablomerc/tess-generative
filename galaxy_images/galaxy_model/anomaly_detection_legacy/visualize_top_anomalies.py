"""
Thin wrapper around anomaly_detection/visualize_top_anomalies.py that redirects
the default output directory to anomaly_detection_legacy/outputs/figures_<suffix>.

The thumbnail panels still come from images_hsc / images_legacy in the same
neighbours_v2.h5 file (so each top-N grid still shows HSC top-row + Legacy
bottom-row), but the *ranking* of which galaxies are most anomalous is driven
by the Legacy-derived scores produced by this folder's fit_and_score.py.
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_hsc_dir = _here.parent / "anomaly_detection"
if str(_hsc_dir) not in sys.path:
    sys.path.insert(0, str(_hsc_dir))

import visualize_top_anomalies as _orig


def main():
    # Patch the module-level _here used by argument-default resolution and the
    # implicit out_dir = _here/"outputs"/"figures_<suffix>" path.
    _orig._here = _here
    _orig.main()


if __name__ == "__main__":
    main()
