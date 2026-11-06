"""
Навчання на правках користувача і злиття з датчиком: перевірки з чесними очікуваннями.
"""
import csv
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydrobrain.assimilation import SoilMoistureKalman  # noqa: E402
from hydrobrain.learning import FeatureEngineer, UserPreferenceLearner  # noqa: E402
from test_hydrobrain import NOW, _brain  # noqa: E402


# ----------------------------------------------------------------- множник користувача
def _simulate(user, factor, n=25, seed=0):
    """user='consistent' — людина ллє factor × потреба (свою звичну норму);
    user='anchored' — людина ллє factor × показане системою (завжди «трохи більше, ніж сказали»)."""
    L = UserPreferenceLearner(FeatureEngineer())
    rng = np.random.default_rng(seed)
    mults = []
    for k in range(n):
        ctx = dict(dr_rel=0.5 + 0.1 * rng.normal(), eto_3d=4 + rng.normal(), doy=150 + 3 * k, hour=7)
        m = L.multiplier(ctx)
        base = 20.0
        shown = base * m
        applied = factor * (base if user == "consistent" else shown)
        L.observe_amount(ctx, base, applied)
        mults.append(m)
    return np.array(mults), L


def test_consistent_user_is_learned_without_oscillation():
    m, _ = _simulate("consistent", 1.2)
    assert abs(m[-1] - 1.2) < 0.03                 # вивчив звичку (стара версія зупинялась на ~×1.1)
    assert np.all(np.diff(m[1:]) > -0.02)          # без коливань «вгору-вниз»


def test_user_can_teach_less_water():
    m, _ = _simulate("consistent", 0.7)
    assert 0.65 < m[-1] < 0.78


def test_overwatering_is_capped():
    m, L = _simulate("anchored", 1.3)
    assert m.max() <= L.max_up + 1e-9               # без обмеження множник ріс би до ×2


def test_log_irrigation_learns_relative_to_base_norm(tmp_path):
    b, fid = _brain(tmp_path)
    f = b.load_field(fid)
    f["last_recommendation"] = dict(time=NOW.isoformat(), action="irrigate_now", gross_mm=25.0, user_mult=1.25,
                                    ctx=dict(dr_rel=0.5, doy=196, hour=10, dr=40.0, raw=35.0))
    b._save(f)
    res = b.log_irrigation(fid, 25.0, time=NOW + timedelta(minutes=30))
    assert abs(res["learned"]["base"] - 20.0) < 1e-6      # показані 25 мм = 20 мм моделі × 1.25
    rows = list(csv.DictReader(open(tmp_path / "training_log.csv", encoding="utf-8")))
    assert abs(float(rows[-1]["y_amount_logratio"]) - math.log(1.25)) < 1e-6


# ----------------------------------------------------------------- зміщення датчика
def _kalman_run(pin, bias=0.02, every=12, seed=0, days=40):
    """Модель стартує з помилкою дефіциту 15 мм; раз на 10 діб злива 40 мм насичує ґрунт до FC."""
    rng = np.random.default_rng(seed)
    zr, fc, taw = 0.5, 0.32, 80.0
    dr_true = 30.0
    k = SoilMoistureKalman(15.0, 10.0, pin_at_fc=pin)

    def step(dr, et, rain):
        dr = dr + et - rain
        dp = 0.0
        if dr < 0:
            dp = min(-dr, 2.0)
            dr += dp
        return dr, dp

    for h in range(days * 24):
        et = 0.25 if 6 <= h % 24 <= 18 else 0.0
        rain = 40.0 if h % 240 == 200 else 0.0
        dr_true, _ = step(dr_true, et, rain)
        dr_model, dp = step(k.dr, et, rain)
        k.predict(dr_model, dict(et=et, t=0.8 * et, pe=rain, inet=0.0, dp=dp, taw=taw))
        if h % every == 0 and h > 0:
            z = fc - dr_true / (1000 * zr) + bias + rng.normal(0, 0.01)
            k.update(z, zr, fc, 0.01, 0.01, dr_bounds=(-50, taw))
    return k.bias, abs(k.dr - dr_true)


def test_sensor_bias_needs_fc_anchor():
    """Чесне очікування: з одним датчиком зміщення стає помітним лише після насичення ґрунту до FC.
    Без «якоря» фільтр не відрізняє зміщення датчика від помилки дефіциту (оцінка ≈ 0)."""
    off = [_kalman_run(False, seed=s) for s in range(8)]
    on = [_kalman_run(True, seed=s) for s in range(8)]
    bias_off = np.mean([x[0] for x in off])
    bias_on = np.mean([x[0] for x in on])
    assert abs(bias_off) < 0.006                     # без якоря зміщення майже не видно
    assert bias_on > 0.4 * 0.02                      # з якорем — знайдено помітну частину (не все!)
    assert np.mean([x[1] for x in on]) < np.mean([x[1] for x in off])   # дефіцит оцінено точніше


# ----------------------------------------------------------------- калібрування θFC / θWP
def _synthetic_sensor(fc=0.25, wp=0.08, days=120, seed=1):
    """Ґрунт, що після кожного дощу стікає до fc за ~1 добу, а між дощами висихає до ~wp."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-04-01", periods=days * 24, freq="1h", tz="UTC")
    th = np.empty(len(t))
    rain = np.zeros(len(t))
    x = 0.15
    for i in range(len(t)):
        if i % (24 * 12) == 100:              # злива раз на 12 діб
            rain[i] = 30.0
            x = fc + 0.08
        if x > fc:
            x = fc + (x - fc) * 0.9            # швидке стікання надлишку
        else:
            x = max(wp, x - 0.0012 * (x - wp) / (fc - wp) * 3)   # висихання, повільнішає біля wp
        th[i] = x + rng.normal(0, 0.002)
    return t, th, rain


def test_fc_wp_estimated_from_sensor():
    from hydrobrain.soil_calibration import estimate_fc_wp
    t, th, rain = _synthetic_sensor()
    r = estimate_fc_wp(t, th, prior_fc=0.13, prior_wp=0.10, rain_times=t, rain_mm=rain)
    assert r["n_events"] >= 3
    assert abs(r["fc_from_data"] - 0.25) < 0.02             # знайшли справжню FC
    assert 0.13 < r["theta_fc"] <= 0.26                       # з урахуванням довідника — зсунулись до правди
    assert r["theta_wp"] < 0.10                               # ґрунт висихав нижче довідника → θWP знижено


def test_fc_wp_keeps_prior_without_evidence():
    from hydrobrain.soil_calibration import estimate_fc_wp
    import pandas as pd
    t = pd.date_range("2026-06-01", periods=24 * 10, freq="1h", tz="UTC")
    th = np.linspace(0.30, 0.27, len(t))                     # 10 діб повільного висихання, без дощів
    r = estimate_fc_wp(t, th, prior_fc=0.32, prior_wp=0.15)
    assert r["n_events"] == 0 and r["theta_fc"] == 0.32 and r["theta_wp"] == 0.15


def test_brain_calibrate_soil_updates_params_and_replays_history(tmp_path):
    b, fid = _brain(tmp_path)
    t, th, rain = _synthetic_sensor(fc=0.30, wp=0.12, days=60)
    for k in range(0, len(t), 3):                              # показ кожні 3 год
        b.add_sensor_reading(fid, float(th[k]), depth_cm=30, time=t[k].to_pydatetime())
    b.log_irrigation(fid, 20.0, time=t[500].to_pydatetime())
    f0 = b.load_field(fid)
    res = b.calibrate_soil(fid, weather=dict(times=list(t), precipitation=rain))
    f = b.load_field(fid)
    assert res["applied"] and f["provenance"]["theta_fc"]["source"] == "sensor"
    assert abs(f["params"]["theta_fc"] - f0["soil_hydraulics"]["theta_fc"]) > 0.005
    assert f["state"] is None and not any(e.get("applied") for e in f["events"]["irrigation"])
    # повторне оновлення параметрів (не ґрунту) зберігає калібрування
    b.update_params(fid, {"irrigation_method": "sprinkler"})
    assert b.load_field(fid)["provenance"]["theta_fc"]["source"] == "sensor"


# ----------------------------------------------------------------- ML-ядро прогнозу (1.6.0)
def _field_with_sensor(tmp_path, theta=0.33, hours=(1, 13, 25, 37, 49, 73)):
    b, fid = _brain(tmp_path)
    for hh in hours:
        b.add_sensor_reading(fid, theta, depth_cm=30, time=NOW - timedelta(hours=hh))
    return b, fid


def test_ml_core_is_used_with_fresh_sensor(tmp_path):
    from test_hydrobrain import _wx
    b, fid = _field_with_sensor(tmp_path)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    fc = r["forecast_core"]
    assert fc["core"].startswith("ml") and fc["target"] in ("T2", "T3", "RZ")
    assert np.isfinite(fc["shift_mm"]["24h"]) and fc["sigma_mm"]["48h"] > 0
    assert "_shift" not in fc and "_log" not in fc                     # службові ключі не потрапляють у відповідь
    assert len(b.load_field(fid)["ml_log"]) == 1


def test_ml_core_falls_back_to_physics(tmp_path, monkeypatch):
    from test_hydrobrain import _wx
    from hydrobrain import ml_forecast as mlf
    b, fid = _brain(tmp_path / "a")                                     # без датчика
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["forecast_core"]["core"] == "physics" and "датчика" in r["forecast_core"]["reason"]
    b2, fid2 = _field_with_sensor(tmp_path / "b")                        # моделі недоступні
    monkeypatch.setattr(mlf.MLCore, "load", classmethod(lambda cls, path=None: (None, "немає моделі")))
    r2 = b2.recommend(fid2, now=NOW, weather=_wx([]))
    assert r2["forecast_core"]["core"] == "physics" and r2["action"] in ("skip", "postpone", "irrigate_now")


def test_daily_weather_and_scale():
    from datetime import datetime, timezone
    from hydrobrain import ml_forecast as mlf
    now = datetime(2026, 7, 10, 0, tzinfo=timezone.utc)
    times = [now + timedelta(hours=k) for k in range(-200, 72)]
    n = len(times)
    d = mlf.daily_weather(times, now, np.ones(n), np.full(n, 0.2), np.full(n, 20.0), np.full(n, 60.0),
                          np.full(n, 300.0), np.full(n, 2.0))
    assert d["P_past1"] == 24 and d["P_past7"] == 168 and d["P_1"] == 24 and abs(d["ET0_2"] - 4.8) < 1e-9
    lo, hi, src = mlf.sensor_scale([now], [0.3], 0.1, 0.35)
    assert (lo, hi) == (0.1, 0.35) and "довідник" in src
    ts = [now + timedelta(days=k) for k in range(10)]
    lo, hi, src = mlf.sensor_scale(ts, [0.30, 0.301] * 5, 0.1, 0.35)
    assert abs((hi - lo) - mlf.MIN_RANGE) < 1e-9 and "датчик" in src


def test_finetune_local_gate(tmp_path):
    b, fid = _field_with_sensor(tmp_path)
    res = b.finetune_local(fid)
    assert res["enabled"] is False                                     # даних поля ще немає
    # синтетична історія поля: справжня зміна стабільно на 0.05 більша за прогноз ядра → поправка має увімкнутись
    from hydrobrain import ml_forecast as mlf
    f = b.load_field(fid)
    rng = np.random.default_rng(0)
    f["events"]["sensor"], f["ml_log"] = [], []
    lo, hi = 0.15, 0.17          # вузький масштаб, щоб θ лишалась у фізичних межах (> 1 трактувалось би як %)
    for k in range(120):
        t0 = NOW - timedelta(days=200 - k)
        feats = {str(h): {c: float(rng.normal()) for c in mlf.feature_names(h)} for h in (1, 2)}
        for h in (1, 2):
            feats[str(h)]["core"] = float(rng.normal(0, 0.02))
        f["ml_log"].append(dict(time=t0.isoformat(), target="T2", lo=lo, hi=hi, w_now=0.5, feats=feats))
    for k in range(122):
        t = NOW - timedelta(days=200 - k)
        f["events"]["sensor"].append(dict(time=t.isoformat(), theta=0.25, depth_cm=30, sensor_id="s1"))
    b._save(f)
    # підганяємо покази так, щоб y = core + 0.05 для горизонту 1 доба
    f = b.load_field(fid)
    th = 0.25
    sens = []
    for k, r in enumerate(f["ml_log"]):
        sens.append(th)
        th = th + (r["feats"]["1"]["core"] + 0.05) * (hi - lo)
    sens.append(th)
    for e, v in zip(f["events"]["sensor"], sens):
        e["theta"] = v
    b._save(f)
    res = b.finetune_local(fid)
    assert res["horizons"]["1"]["enabled"] and res["horizons"]["1"]["mae_local_val"] < res["horizons"]["1"]["mae_core_val"]


# ----------------------------------------------------------------- оцінка стану без датчика (1.7.0)
def _fake_era5l(lat, lon, start, end):
    from datetime import datetime, timezone
    import pandas as pd
    t = pd.date_range(pd.Timestamp(start).tz_localize("UTC"), pd.Timestamp(end).tz_localize("UTC") + pd.Timedelta(hours=23),
                      freq="1h")
    doy = t.dayofyear.to_numpy()
    base = 0.30 - 0.08 * np.sin(2 * np.pi * (doy - 100) / 365)        # сезонне висихання
    return dict(times=[x.to_pydatetime() for x in t], sm1=base - 0.02, sm2=base, sm3=base + 0.03)


def test_no_sensor_uses_reanalysis_state_once_per_day(tmp_path, monkeypatch):
    from test_hydrobrain import _wx
    b, fid = _brain(tmp_path)
    monkeypatch.setattr(b.sources, "era5land_soil", _fake_era5l)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    fc = r["forecast_core"]
    assert fc["state_source"] == "reanalysis" and fc["core"] == "ml" and fc["state"]["applied"]
    assert any("ERA5-Land" in d for d in r["details"])
    assert "era5l_clim" in b.load_field(fid)                              # кліматологія точки закешована
    r2 = b.recommend(fid, now=NOW + timedelta(hours=1), weather=_wx([]))
    assert r2["forecast_core"]["state"]["applied"] is False               # той самий «вимір» не вдруге за добу


def test_stale_sensor_is_rolled_forward(tmp_path, monkeypatch):
    from test_hydrobrain import _wx
    b, fid = _field_with_sensor(tmp_path, hours=(5 * 24, 6 * 24, 8 * 24))
    monkeypatch.setattr(b.sources, "era5land_soil", _fake_era5l)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    st = r["forecast_core"]
    assert st["state_source"] == "sensor_stale" and 4.5 < st["state"]["lag_days"] < 5.5
    assert any("Останній показ датчика" in d for d in r["details"])
