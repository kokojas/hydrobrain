"""
Автотести (pytest). Запуск:  python -m pytest -q tests/
Перевіряють формули на офіційних прикладах FAO-56, парсинг API (на зразках відповідей) і логіку рішень.
"""
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydrobrain import HydroBrain, meteo, soil, crop as cropm, salinity  # noqa: E402
from hydrobrain.databases import CROPS  # noqa: E402
from hydrobrain.sources import DataSources  # noqa: E402
from hydrobrain.synthetic import make_weather  # noqa: E402


# ----------------------------------------------------------------- FAO-56
def test_eto_daily_fao56_example18():
    # Уккль (Брюссель), 6 липня: ETo = 3.9 мм/добу
    e = meteo.eto_daily(21.5, 12.3, 22.07, 2.078, 50.8, 187, 100, rhmax=84, rhmin=63)
    assert abs(e - 3.9) < 0.05


def test_eto_hourly_fao56_example19():
    t = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)   # 14-15 год місцевого часу, Сенегал
    ra, _, _ = meteo.ra_hourly(16.217, -16.25, t)
    assert abs(ra - 3.543) < 0.01
    eto, _, rn = meteo.eto_hourly(38, 52, 3.3, 2.450, meteo.pressure_from_altitude(8), ra, 8, True)
    assert abs(eto - 0.63) < 0.01 and abs(rn - 1.749) < 0.01
    t2 = datetime(2026, 10, 1, 4, 0, tzinfo=timezone.utc)
    ra2, _, _ = meteo.ra_hourly(16.217, -16.25, t2)
    eto2, _, rn2 = meteo.eto_hourly(28, 90, 1.9, 0.0, meteo.pressure_from_altitude(8), ra2, 8, False, 0.8)
    assert eto2 < 0.01 and abs(rn2 + 0.100) < 0.01


def test_runoff_cn_known_value():
    # CN=80, P=100 мм: S=63.5, Ia=12.7 → Q=(87.3²)/(150.8)=50.5 мм
    q = float(soil.runoff_cn(100.0, 80.0))
    assert abs(q - 50.5) < 0.3


def test_saxton_rawls_loam():
    h = soil.saxton_rawls(40, 20, 2.5)
    assert 0.25 < h["theta_fc"] < 0.33 and 0.10 < h["theta_wp"] < 0.16


def test_beta_development_peak_at_to():
    tb, to, tc = CROPS["maize"]["t_card"]
    t = np.linspace(tb, tc, 500)
    f = cropm.beta_rate(t, tb, to, tc)
    assert abs(t[np.argmax(f)] - to) < 0.2 and f.max() == pytest.approx(1.0, abs=1e-3)


def test_salinity_ks():
    assert salinity.ks_salinity(1.0, 1.7, 12, 1.25) == 1.0
    assert salinity.ks_salinity(3.7, 1.7, 12, 1.25) == pytest.approx(1 - 12 / 125 * 2.0)


def test_slope_radiation_south_vs_north():
    t = datetime(2026, 6, 21, 10, 0, tzinfo=timezone.utc)
    ra, _, _ = meteo.ra_hourly(49.0, 30.0, t)
    rs = 0.7 * ra
    s = meteo.slope_radiation_factor(49, 30, t, rs, ra, slope_deg=20, aspect_deg=180)
    n = meteo.slope_radiation_factor(49, 30, t, rs, ra, slope_deg=20, aspect_deg=0)
    assert s > 1.0 > n


# ----------------------------------------------------------------- парсинг API (зразки відповідей)
def test_parse_open_meteo_sample():
    ts = [1781568000 + 3600 * i for i in range(3)]
    data = {"elevation": 180.0, "hourly": {"time": ts, "temperature_2m": [20, 21, None], "precipitation": [0, 0.2, 1.0]}}
    out = DataSources._parse_hourly(data)
    assert out["times"][0].tzinfo is not None and np.isnan(out["temperature_2m"][2])


def test_terrain_from_dem(tmp_path, monkeypatch):
    ds = DataSources(str(tmp_path), offline=False)
    # площина, що підіймається на північ на 9 м/90 м → ухил ≈ 5.7°, схил дивиться на південь (180°)
    z = [109, 109, 109, 100, 100, 100, 91, 91, 91]
    monkeypatch.setattr(ds, "_get", lambda url, params, ttl_s=0, binary=False, expect_tiff=False: {"elevation": z})
    t = ds.terrain(49.0, 30.0)
    assert abs(t["slope_deg"] - 5.71) < 0.1 and abs(t["aspect_deg"] - 180) < 1


def test_geology_keywords(tmp_path, monkeypatch):
    ds = DataSources(str(tmp_path))
    resp = {"results": [{"attributes": {"Lithology": "Loess, silt", "Age": "Pleistocene"}}]}
    monkeypatch.setattr(ds, "_get", lambda url, params, ttl_s=0, binary=False, expect_tiff=False: resp)
    assert ds.geology(49.0, 30.0)["geo"] == "loess"


def test_ensemble_parse(tmp_path, monkeypatch):
    ds = DataSources(str(tmp_path))
    ts = [1781568000 + 3600 * i for i in range(4)]
    resp = {"hourly": {"time": ts, "precipitation": [0, 1, 0, 0], "precipitation_member01": [0, 2, 0, 0],
                       "precipitation_member02": [0, 0, None, 0]}}
    monkeypatch.setattr(ds, "_get", lambda url, params, ttl_s=0, binary=False, expect_tiff=False: resp)
    e = ds.ensemble_precip(49, 30)
    assert e["members"].shape == (3, 4)


# ----------------------------------------------------------------- логіка рішень
NOW = datetime(2026, 7, 15, 7, 0, tzinfo=timezone.utc)


def _brain(tmp_path, **extra):
    b = HydroBrain(str(tmp_path), offline=True)
    inp = dict(lat=49.23, lon=28.47, crop="maize", sowing_date="2026-05-01", irrigation_method="drip",
               soil_type_ua="chornozem_typical", area_m2=10000)
    inp.update(extra)
    fid = b.create_field(inp)["field_id"]
    return b, fid


def _wx(future_rain, days_back=60):
    start = NOW - timedelta(days=days_back)
    hp = days_back * 24
    ev = [(hp + h, mm, d, p) for h, mm, d, p in future_rain]
    return make_weather(49.23, 28.47, start, hp + 96, rain_events=ev)


def test_dry_soil_means_irrigate(tmp_path):
    b, fid = _brain(tmp_path)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["action"] == "irrigate_now" and r["amount"]["gross_mm"] > 0
    # поле без датчика: з 1.8 рекомендація орієнтовна (див. test_no_sensor_policy.py)
    assert r["orientative"] and "потрібен полив" in r["summary"]


def test_beginner_cannot_override_expert_params(tmp_path):
    b = HydroBrain(str(tmp_path), offline=True)
    f = b.create_field(dict(lat=49, lon=30, crop="tomato", sowing_date="2026-05-01", bd=1.7), mode="beginner")
    assert f["params"]["bd"] != 1.7 and any("розширеному" in w for w in f["warnings"])
    f2 = b.create_field(dict(lat=49, lon=30, crop="tomato", sowing_date="2026-05-01", bd=1.7), mode="expert")
    assert f2["params"]["bd"] == 1.7


def test_user_feedback_learning(tmp_path):
    b, fid = _brain(tmp_path)
    for k in range(6):
        now = NOW + timedelta(days=k)
        r = b.recommend(fid, now=now, weather=make_weather(49.23, 28.47, NOW - timedelta(days=60), 70 * 24))
        if r["amount"]:
            b.log_irrigation(fid, r["amount"]["gross_mm"] * 1.4, time=now + timedelta(minutes=20))
    f = b.load_field(fid)
    assert f["learners"]["user"]["rls"]["n"] >= 2
    r = b.recommend(fid, now=NOW + timedelta(days=6, hours=1),
                    weather=make_weather(49.23, 28.47, NOW - timedelta(days=60), 70 * 24))
    if r["amount"]:
        assert r["amount"]["user_multiplier"] > 1.05


def test_soilgrids_wcs_urban_center_uses_neighbors(tmp_path, monkeypatch):
    import io
    import tifffile
    ds = DataSources(str(tmp_path))
    vals = {"sand": 180, "clay": 300, "silt": 520, "soc": 250, "bdod": 125, "phh2o": 68, "cfvo": 20}

    def fake_get(url, params, ttl_s=0, binary=False, expect_tiff=False):
        prop = params["COVERAGEID"].split("_")[0]
        a = np.full((17, 17), vals[prop], dtype=np.int16)
        a[5:12, 5:12] = -32768              # «місто» в центрі — nodata
        buf = io.BytesIO()
        tifffile.imwrite(buf, a)
        return buf.getvalue()
    monkeypatch.setattr(ds, "_get", fake_get)
    monkeypatch.setattr(ds, "_soilgrids_rest", lambda la, lo: (_ for _ in ()).throw(RuntimeError("paused")))
    r = ds.soilgrids(49.23, 28.47)
    assert abs(r["sand"] - 18) < 0.01 and abs(r["clay"] - 30) < 0.01 and "note" in r


def test_soilgrids_rest_neighbor(tmp_path, monkeypatch):
    ds = DataSources(str(tmp_path))
    calls = []

    def fake_point(la, lo):
        calls.append((la, lo))
        return {} if len(calls) == 1 else {"sand": 20.0, "clay": 28.0, "soc": 20.0}
    monkeypatch.setattr(ds, "_sg_rest_point", fake_point)
    r = ds.soilgrids(49.23, 28.47)
    assert len(calls) == 2 and r["source"] == "SoilGrids REST" and "note" in r
