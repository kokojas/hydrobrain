"""
Перевірки логіки рішень, які мають бути істинними за будь-якої реалізації правил
(монотонність, крайні випадки), а не лише «програма щось вивела».
Запуск:  python -m pytest -q tests/
"""
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_hydrobrain import NOW, _brain, _wx  # noqa: E402

URGENCY = {"skip": 0, "postpone": 1, "irrigate_now": 2}


def _rec(tmp_path, theta, rain=(), **extra):
    # v1.2: модель має два шари (0–0.3 м і нижче), тому для повної картини — два датчики: 20 і 60 см
    b, fid = _brain(tmp_path, **extra)
    for h in (1, 13, 25, 37):
        b.add_sensor_reading(fid, theta, depth_cm=20, time=NOW - timedelta(hours=h))
        b.add_sensor_reading(fid, theta, depth_cm=60, time=NOW - timedelta(hours=h), sensor_id="s2")
    return b.recommend(fid, now=NOW, weather=_wx(list(rain)))


def test_wet_soil_is_skip_with_low_risk(tmp_path):
    # v1.2: чорнозем типовий має виміряні константи [Д65] θFC = 0.408, θWP = 0.235 → «вологий» = близько FC
    r = _rec(tmp_path, 0.40)
    assert r["action"] == "skip"
    assert r["moisture"]["available_water_pct"] > 80
    assert r["risk"]["p_stress_48h"] < 0.25


def test_very_dry_soil_is_irrigate_now(tmp_path):
    r = _rec(tmp_path, 0.25)
    assert r["action"] == "irrigate_now" and r["amount"]["gross_mm"] > 0
    assert any(x["code"] == "critical_deficit" for x in r["reasons"])


def test_drier_soil_is_never_less_urgent(tmp_path):
    thetas = [0.37, 0.34, 0.31, 0.28, 0.25]
    recs = [_rec(tmp_path / f"t{i}", th) for i, th in enumerate(thetas)]
    urg = [URGENCY[r["action"]] for r in recs]
    assert urg == sorted(urg), f"терміновість має не спадати з висиханням: {list(zip(thetas, urg))}"
    p = [r["risk"]["p_stress_48h"] for r in recs]
    assert p == sorted(p), f"ймовірність стресу має не спадати: {p}"


@pytest.mark.parametrize("theta", [0.33, 0.31])
def test_forecast_rain_never_increases_irrigation(tmp_path, theta):
    dry = _rec(tmp_path / "dry", theta)
    wet = _rec(tmp_path / "wet", theta, rain=[(6, 8.0, 3, 95)])     # ~24 мм через 6 год, 95%
    assert URGENCY[wet["action"]] <= URGENCY[dry["action"]]
    if dry["amount"] and wet["amount"]:
        assert wet["amount"]["gross_mm"] <= dry["amount"]["gross_mm"] + 1e-6


def test_off_season_is_skip(tmp_path):
    r = _rec(tmp_path, 0.25, sowing_date=(NOW + timedelta(days=20)).date().isoformat())
    assert r["action"] == "skip" and any(x["code"] == "off_season" for x in r["reasons"])


def test_decision_thresholds_can_be_overridden(tmp_path):
    b, fid = _brain(tmp_path)
    for h in (1, 13, 25, 37):
        b.add_sensor_reading(fid, 0.33, depth_cm=30, time=NOW - timedelta(hours=h))
    f = b.load_field(fid)
    f["params"]["decision_thresholds"] = {"p_skip": 0.0}     # «ніколи не пропускати, якщо стрес можливий»
    b._save(f)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["action"] in ("irrigate_now", "postpone")
