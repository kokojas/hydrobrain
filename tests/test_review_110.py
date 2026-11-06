"""
1.10: виправлення за рецензією — кожен тест відтворює приклад рецензента.
  1) норма і інфільтрація при дощуванні; 2) донавчання не вчиться на поливі;
  3) датчики різної глибини → один профіль; 4) однакова нормалізація профілів у навчанні й застосунку.
"""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydrobrain import HydroBrain  # noqa: E402
from hydrobrain import ml_forecast as mlf  # noqa: E402
from hydrobrain.decision import infiltration_plan  # noqa: E402
from test_hydrobrain import NOW, _wx  # noqa: E402

INP = dict(lat=49.23, lon=28.47, crop="maize", sowing_date="2026-05-01", soil_type_ua="chornozem_typical",
           area_m2=10000)


# ----------------------------------------------------------------- 1. інфільтрація
def test_infiltration_plan_formula():
    p = infiltration_plan(20.0, 20.0, 2.0, 3.0, 8)
    assert p["on_min"] == 10 and p["soak_min"] == 90 and abs(p["per_cycle_mm"] - 3.3) < 0.05
    assert p["cycles"] == 6 and not p["limited"] and p["continuous_infiltration"] == 0.1


def _sprinkler(tmp_path, **extra):
    b = HydroBrain(str(tmp_path), offline=True, no_sensor_policy="full")
    fid = b.create_field(dict(INP, irrigation_method="sprinkler", application_rate_mm_h=20, ksat_mm_h=2, **extra),
                         mode="expert")["field_id"]
    for h in (1, 25):
        b.add_sensor_reading(fid, 0.27, depth_cm=20, time=NOW - timedelta(hours=h))
    return b, fid


def test_norm_is_consistent_with_infiltration(tmp_path):
    b, fid = _sprinkler(tmp_path)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    am = r["amount"]
    codes = [x["code"] for x in r["reasons"]]
    assert am["cycles"]["cycles"] >= 2 and am["cycles"]["rate_mm_h"] == 20.0
    assert "infiltration_cycles" in codes or "infiltration_limited" in codes
    assert "reduced_for_rain" not in codes                     # дощу немає — і пояснення не про дощ
    # нетто = те, що реально вбереться за планом циклів (ефективність способу, без 90% стоку)
    assert abs(am["net_mm"] - am["gross_mm"] * am["efficiency"]) < 1.0
    assert "циклами" in r["headline"]
    # без циклів у ґрунт увійшло б лише Ksat / I = 10% — саме це раніше «губилось»
    assert am["cycles"]["continuous_infiltration"] == 0.1


def test_logged_irrigation_follows_cycle_plan(tmp_path):
    b, fid = _sprinkler(tmp_path)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    b.log_irrigation(fid, r["amount"]["gross_mm"], time=NOW + timedelta(hours=1))
    assert b.load_field(fid)["events"]["irrigation"][-1]["cycles"] is True
    b.log_irrigation(fid, 10.0, time=NOW + timedelta(hours=60), cycles=False)
    assert b.load_field(fid)["events"]["irrigation"][-1]["cycles"] is False


# ----------------------------------------------------------------- 2. донавчання без поливу
def _synthetic_history(b, fid, n=120):
    f = b.load_field(fid)
    rng = np.random.default_rng(0)
    f["ml_log"], f["events"]["sensor"] = [], []
    th = 0.25
    for k in range(n + 1):
        t = NOW - timedelta(days=200 - k)
        f["events"]["sensor"].append(dict(time=t.isoformat(), theta=th, depth_cm=30, sensor_id="s1"))
        if k < n:
            feats = {str(h): {c: float(rng.normal()) for c in mlf.feature_names(h)} for h in (1, 2)}
            for h in (1, 2):
                feats[str(h)]["core"] = float(rng.normal(0, 0.02))
            f["ml_log"].append(dict(time=t.isoformat(), target="T2", lo=0.15, hi=0.17, w_now=0.5, zr=0.3,
                                    feats=feats))
            th += (feats["1"]["core"] + 0.05) * 0.02
    b._save(f)


def test_local_rows_exclude_irrigation(tmp_path):
    b = HydroBrain(str(tmp_path), offline=True)
    fid = b.create_field(dict(INP, irrigation_method="drip"))["field_id"]
    _synthetic_history(b, fid)
    n0 = len(b._local_rows(b.load_field(fid), None, 1))
    for d in (150, 120, 90):                                  # три поливи посеред інтервалів
        b.log_irrigation(fid, 20.0, time=NOW - timedelta(days=d, hours=-6))
    n1 = len(b._local_rows(b.load_field(fid), None, 1))
    assert n0 - n1 >= 3                                       # інтервали з поливом не стали мітками
    n2 = len(b._local_rows(b.load_field(fid), None, 2))
    assert n2 <= n1 - 3                                       # для 48 год виключається ще більше


def test_irrigation_masks_past_change_features(tmp_path):
    b = HydroBrain(str(tmp_path), offline=True)
    fid = b.create_field(dict(INP, irrigation_method="drip"))["field_id"]
    for h in (1, 25, 73):
        b.add_sensor_reading(fid, 0.30, depth_cm=20, time=NOW - timedelta(hours=h))
    b.log_irrigation(fid, 15.0, time=NOW - timedelta(hours=10))
    b.recommend(fid, now=NOW, weather=_wx([]))
    feats = b.load_field(fid)["ml_log"][-1]["feats"]["1"]
    assert feats["dw1"] is None and feats["dw3"] is None      # зміна містила полив — ознаку не подаємо


# ----------------------------------------------------------------- 3. глибини датчиків
def test_multi_depth_sensors_form_one_profile(tmp_path):
    b = HydroBrain(str(tmp_path), offline=True)
    fid = b.create_field(dict(INP, irrigation_method="drip"))["field_id"]
    b.add_sensor_reading(fid, 0.20, depth_cm=20, time=NOW, sensor_id="a")
    b.add_sensor_reading(fid, 0.40, depth_cm=40, time=NOW, sensor_id="b")
    reads = b._sensor_readings(b.load_field(fid), b.load_field(fid)["params"], zr=0.6)
    assert len(reads) == 1 and abs(reads[0][1] - 0.30) < 1e-9   # не 0.20 і не два «ряди» в одному
    series = {0.2: [(NOW, 0.2)], 0.6: [(NOW, 0.5)]}
    assert abs(mlf.build_profile(series, 0.3)[0][1] - 0.2) < 1e-9  # датчик глибше коренів не враховано


# ----------------------------------------------------------------- 4. нормалізація профілів
def test_profile_normalization_aggregates_theta_first(monkeypatch):
    from ds.stage2 import data_io as io
    t = pd.date_range("2016-04-01", periods=24 * 60, freq="1h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"time": t, "year": t.year,
                       io.SOIL_COLS[0]: 0.10 + 0.20 * rng.random(len(t)),
                       io.SOIL_COLS[1]: 0.25 + 0.05 * rng.random(len(t)),
                       io.SOIL_COLS[2]: 0.30 + 0.01 * rng.random(len(t))})
    monkeypatch.setattr(io, "load_hourly", lambda name: df)
    lp = io.layer_profiles("x", years=[2016])
    th = io.profile_theta(df, "T2").to_numpy()
    lo, hi = np.nanpercentile(th, 1), np.nanpercentile(th, 99)
    expect = io.to_rel(th, (lo, hi))[df["time"].dt.hour.to_numpy() == 0]
    assert np.allclose(lp["w_T2"].to_numpy(), expect)
    # і це справді інша величина, ніж середнє окремо нормалізованих шарів (як було до 1.10)
    rel = [io.to_rel(df[c], (np.percentile(df[c], 1), np.percentile(df[c], 99))) for c in io.SOIL_COLS]
    old = ((7 * rel[0] + 21 * rel[1] + 2 * rel[2]) / 30)[df["time"].dt.hour.to_numpy() == 0]
    assert np.max(np.abs(old - expect)) > 0.05


def test_core_model_is_v2():
    core, _ = mlf.MLCore.load()
    assert core is not None and core.meta["version"] == "ml_core_v2"
