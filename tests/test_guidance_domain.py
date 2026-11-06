"""
1.9: панель «Як система рахує це поле», автоматичне донавчання, перевірка домену ML-ядра.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydrobrain import HydroBrain  # noqa: E402
from hydrobrain import ml_forecast as mlf  # noqa: E402
from test_hydrobrain import NOW, _wx  # noqa: E402
from test_learning import _fake_era5l  # noqa: E402

INP = dict(lat=49.23, lon=28.47, crop="maize", sowing_date="2026-05-01", irrigation_method="drip",
           soil_type_ua="chornozem_typical", area_m2=10000)


def _field(tmp_path, sensor_hours=(), mode="beginner", **extra):
    b = HydroBrain(str(tmp_path), offline=True)
    fid = b.create_field(dict(INP, **extra), mode=mode)["field_id"]
    for h in sensor_hours:
        b.add_sensor_reading(fid, 0.30, depth_cm=20, time=NOW - timedelta(hours=h))
    return b, fid


def test_guidance_no_sensor_and_learning(tmp_path):
    b, fid = _field(tmp_path / "a")
    g = b.recommend(fid, now=NOW, weather=_wx([]))["guidance"]
    assert g["stage"] == "no_sensor" and "датчика" in g["next_step"]
    b2, fid2 = _field(tmp_path / "b", sensor_hours=(1, 25))
    g2 = b2.recommend(fid2, now=NOW, weather=_wx([]))["guidance"]
    assert g2["stage"] == "learning" and "із 60" in g2["message"] and "автоматично" in g2["next_step"]


def test_guidance_stale_sensor(tmp_path, monkeypatch):
    b, fid = _field(tmp_path, sensor_hours=(5 * 24, 6 * 24))
    monkeypatch.setattr(b.sources, "era5land_soil", _fake_era5l)
    g = b.recommend(fid, now=NOW, weather=_wx([]))["guidance"]
    assert g["stage"] == "stale_sensor" and "5 діб" in g["message"]


def test_domain_check_outside_ukraine(tmp_path):
    paris = dict(lat=48.85, lon=2.35)
    b, fid = _field(tmp_path / "auto", sensor_hours=(1, 25), mode="expert", **paris)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["forecast_core"]["core"] == "physics" and r["forecast_core"]["off_cause"] == "domain"
    assert r["guidance"]["stage"] == "physics_only" and not r["forecast_core"]["domain"]["in_ukraine"]
    b2, fid2 = _field(tmp_path / "ml", sensor_hours=(1, 25), mode="expert", forecast_core="ml", **paris)
    r2 = b2.recommend(fid2, now=NOW, weather=_wx([]))
    assert r2["forecast_core"]["core"].startswith("ml") and "поза умовами навчання" in r2["forecast_core"]["domain_note"]
    assert "Увага" in r2["guidance"]["message"]


def test_user_chosen_physics_and_default_auto(tmp_path):
    b, fid = _field(tmp_path, sensor_hours=(1, 25), mode="expert", forecast_core="physics")
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["forecast_core"]["off_cause"] == "user" and r["guidance"]["stage"] == "physics_only"
    spec = {p["key"]: p for p in HydroBrain.input_schema("expert")["fields"]} \
        if isinstance(HydroBrain.input_schema("expert"), dict) else {p["key"]: p for p in HydroBrain.input_schema("expert")}
    assert spec["forecast_core"]["default"] == "auto"


def test_auto_finetune_runs_when_enough_field_data(tmp_path):
    b, fid = _field(tmp_path)
    f = b.load_field(fid)
    rng = np.random.default_rng(0)
    lo, hi = 0.15, 0.17
    f["ml_log"], f["events"]["sensor"] = [], []
    th = 0.25
    for k in range(121):
        t = NOW - timedelta(days=200 - k)
        f["events"]["sensor"].append(dict(time=t.isoformat(), theta=th, depth_cm=30, sensor_id="s1"))
        if k < 120:
            feats = {str(h): {c: float(rng.normal()) for c in mlf.feature_names(h)} for h in (1, 2)}
            for h in (1, 2):
                feats[str(h)]["core"] = float(rng.normal(0, 0.02))
            f["ml_log"].append(dict(time=t.isoformat(), target="T2", lo=lo, hi=hi, w_now=0.5, feats=feats))
            th += (feats["1"]["core"] + 0.05) * (hi - lo)
    b._save(f)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    loc = b.load_field(fid)["ml_local"]
    assert loc["auto"] and loc["horizons"]["1"]["enabled"] and "event" in r["guidance"]
    # повторно в ту саму добу не перевчається
    r2 = b.recommend(fid, now=NOW + timedelta(hours=2), weather=_wx([]))
    assert "event" not in r2["guidance"]


def test_nearest_reading_bisect():
    t0 = datetime(2026, 7, 1, tzinfo=timezone.utc)
    reads = [(t0 + timedelta(hours=6 * k), float(k)) for k in range(100)]
    assert mlf.nearest_reading(reads, t0 + timedelta(hours=61)) == 10.0
    assert mlf.nearest_reading(reads, t0 + timedelta(hours=64)) is None or \
        mlf.nearest_reading(reads, t0 + timedelta(hours=64)) in (10.0, 11.0)
    assert mlf.nearest_reading(reads, t0 - timedelta(hours=10)) is None
