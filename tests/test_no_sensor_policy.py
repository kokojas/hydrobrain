"""
Поле без датчика (1.8, варіант «Г»): орієнтовна рекомендація, а також перемикач для власника застосунку.
"""
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydrobrain import HydroBrain  # noqa: E402
from test_hydrobrain import NOW, _wx  # noqa: E402
from test_learning import _fake_era5l  # noqa: E402

INP = dict(lat=49.23, lon=28.47, crop="maize", sowing_date="2026-05-01", irrigation_method="drip",
           soil_type_ua="chornozem_typical", area_m2=10000)


def _field(tmp_path, policy, sensor_hours=()):
    b = HydroBrain(str(tmp_path), offline=True, no_sensor_policy=policy)
    fid = b.create_field(dict(INP))["field_id"]
    for h in sensor_hours:
        b.add_sensor_reading(fid, 0.30, depth_cm=20, time=NOW - timedelta(hours=h))
    return b, fid


def test_orientative_without_sensor_keeps_action_but_drops_false_precision(tmp_path):
    bo, fo = _field(tmp_path / "o", "orientative")
    bf, ff = _field(tmp_path / "f", "full")
    ro = bo.recommend(fo, now=NOW, weather=_wx([]))
    rf = bf.recommend(ff, now=NOW, weather=_wx([]))
    assert ro["action"] == rf["action"] == "irrigate_now"            # дія та сама
    assert ro["orientative"] and not rf["orientative"]
    assert ro["confidence_label"] == "низька"
    lo, hi = ro["amount"]["range_mm"]
    assert lo <= ro["amount"]["gross_mm"] <= hi and lo % 5 == 0 and hi % 5 == 0
    assert "орієнтовно" in ro["headline"] and "range_mm" not in rf["amount"]
    assert ro["details"][0].startswith("Це орієнтовна рекомендація без датчика")


def test_policy_off_asks_for_measurement(tmp_path):
    b, fid = _field(tmp_path, "off")
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["action"] == "measure" and r["amount"] is None
    assert b.load_field(fid)["state"] is not None                    # стан поля все одно оновлено


def test_fresh_sensor_is_never_orientative(tmp_path):
    for policy in ("orientative", "off"):
        b, fid = _field(tmp_path / policy, policy, sensor_hours=(1, 25))
        r = b.recommend(fid, now=NOW, weather=_wx([]))
        assert r["orientative"] is False and r["action"] in ("irrigate_now", "postpone", "skip")


def test_stale_sensor_is_not_orientative(tmp_path, monkeypatch):
    b, fid = _field(tmp_path, "orientative", sensor_hours=(4 * 24, 5 * 24))
    monkeypatch.setattr(b.sources, "era5land_soil", _fake_era5l)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["forecast_core"]["state_source"] == "sensor_stale" and r["orientative"] is False


def test_reanalysis_state_is_orientative(tmp_path, monkeypatch):
    b, fid = _field(tmp_path, "orientative")
    monkeypatch.setattr(b.sources, "era5land_soil", _fake_era5l)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["forecast_core"]["state_source"] == "reanalysis" and r["orientative"]
    assert r["confidence_label"] == "низька"


def test_invalid_policy_rejected(tmp_path):
    with pytest.raises(ValueError):
        HydroBrain(str(tmp_path), offline=True, no_sensor_policy="maybe")


def test_user_texture_is_not_overridden_by_default_handbook_constants(tmp_path):
    """1.8: без вибраного типу ґрунту і без SoilGrids довідникові θFC/θWP чорнозему не мають
    підмінювати текстуру, яку задав користувач (пісок 87% отримував θFC 0.408)."""
    b = HydroBrain(str(tmp_path), offline=True)
    f = b.create_field(dict(INP, soil_type_ua=None, sand=87, clay=4, om=3.8), mode="expert")
    assert f["soil_hydraulics"]["theta_fc"] < 0.2
    f2 = b.create_field(dict(INP), mode="expert")                    # вибраний чорнозем — довідник лишається
    assert abs(f2["soil_hydraulics"]["theta_fc"] - 0.408) < 0.01
