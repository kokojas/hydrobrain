"""1.11: перекалібрування ймовірності стресу в рішенні."""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydrobrain import HydroBrain  # noqa: E402
from hydrobrain.decision import _pstress_map  # noqa: E402
from test_hydrobrain import NOW, _wx  # noqa: E402

INP = dict(lat=49.23, lon=28.47, crop="maize", sowing_date="2026-05-01", irrigation_method="drip",
           soil_type_ua="chornozem_typical", area_m2=10000)


def test_calibration_maps_are_monotone_and_bounded():
    for kind in ("ml", "physics"):
        x, y = _pstress_map(kind)
        assert np.all(np.diff(x) >= 0) and np.all(np.diff(y) >= -1e-12)
        assert y.min() >= 0 and y.max() <= 1


def _rec(tmp_path, theta, **extra):
    b = HydroBrain(str(tmp_path), offline=True)
    fid = b.create_field(dict(INP, **extra), mode="expert")["field_id"]
    for h in (1, 25):
        b.add_sensor_reading(fid, theta, depth_cm=20, time=NOW - timedelta(hours=h))
    return b.recommend(fid, now=NOW, weather=_wx([]))


def test_recommendation_reports_raw_and_calibrated_probability(tmp_path):
    r = _rec(tmp_path / "a", 0.33, pstress_calibration=True)
    assert r["risk"]["calibrated"] is True and 0 <= r["risk"]["p_stress_48h_raw"] <= 1
    r2 = _rec(tmp_path / "b", 0.33, pstress_calibration=False)
    assert r2["risk"]["calibrated"] is False
    assert abs(r2["risk"]["p_stress_48h"] - r2["risk"]["p_stress_48h_raw"]) < 1e-9
