"""
Тести виправлень версії 1.2 (див. CHANGELOG.md, розділ [1.2.0]).
Запуск:  python -m pytest -q tests/
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydrobrain import meteo, soil  # noqa: E402
from hydrobrain import crop as cropm  # noqa: E402
from hydrobrain.assimilation import SoilMoistureKalman  # noqa: E402
from hydrobrain.balance import WaterBalanceEngine, moving_mean, Z_UP  # noqa: E402
from hydrobrain.decision import decide  # noqa: E402
from hydrobrain.params import Resolver  # noqa: E402
from hydrobrain.synthetic import make_weather  # noqa: E402

LAT, LON = 49.1, 28.7


def _P(**kw):
    inp = dict(lat=LAT, lon=LON, crop="maize", sowing_date="2026-05-01", irrigation_method="drip",
               soil_type_ua="chornozem_typical")
    mode = "expert" if any(k not in inp for k in kw) else "beginner"
    inp.update(kw)
    P, prov, _ = Resolver(None).resolve(inp, mode)
    return P, prov


def _series(eng, wx, dvs=0.6):
    W = {k: v for k, v in wx.items() if k != "times"}
    d = np.full(len(wx["times"]), dvs)
    return eng.prepare(wx["times"], W, d, W.get("soil_temperature_18cm"))


def _closure(eng, st0, st1, rec):
    """Нев'язка балансу ґрунту: зміна дефіциту мінус сума потоків, з урахуванням росту коренів."""
    growth = 1000 * (eng.hyd["theta_fc"] - eng.theta_sub) * (st1["zr"] - st0["zr"])
    d_store = st1["dr"][0] - st0["dr"][0]
    flux = rec["et"].sum() + rec["dp"].sum() - rec["pe"].sum() - rec["inet"].sum() - rec["cr"].sum()
    return d_store - flux - growth


# ---------------------------------------------------------------- 1. крайовий ефект температури ґрунту
def test_moving_mean_has_no_edge_drop():
    x = np.full(200, 20.0)
    m = moving_mean(x, 72)
    assert np.allclose(m, 20.0)
    old = np.convolve(x, np.ones(72) / 72, mode="same")      # як було у ≤ 1.1
    assert old[-1] < 11.0                                       # стара помилка: падіння ~вдвічі на краю


def test_no_transpiration_drop_at_series_end_without_soil_temp():
    P, _ = _P()
    eng = WaterBalanceEngine(P)
    wx = make_weather(LAT, LON, datetime(2026, 7, 1, tzinfo=timezone.utc), 5 * 24, tmax=27, tmin=15)
    W = {k: v for k, v in wx.items() if k not in ("times", "soil_temperature_18cm")}
    S = eng.prepare(wx["times"], W, np.full(len(wx["times"]), 0.6), None)
    assert S.kx[-1] == pytest.approx(S.kx[len(S.kx) // 2], rel=0.02)


# ---------------------------------------------------------------- 2. сніг і мерзлий ґрунт
def test_snow_accumulates_then_melts_and_mass_closes():
    P, _ = _P(crop="winter_wheat", sowing_date="2025-10-01")
    eng = WaterBalanceEngine(P)
    start = datetime(2026, 1, 10, tzinfo=timezone.utc)
    cold = make_weather(LAT, LON, start, 10 * 24, tmax=-3, tmin=-10, rain_events=[(24, 2.0, 10, 90)])
    S = _series(eng, cold, dvs=0.3)
    st = eng.init_state(1, 20.0, zr0=S.zr[0])
    st0 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    st, rec = eng.run(S, st, cold["precipitation"], np.zeros(len(S.times)))
    assert st["swe"][0] == pytest.approx(20.0, abs=0.01)          # усі 20 мм лягли снігом
    assert rec["pe"].sum() == pytest.approx(0.0, abs=1e-9)        # у ґрунт нічого не потрапило
    assert abs(_closure(eng, st0, st, rec)) < 1e-6
    warm = make_weather(LAT, LON, start + timedelta(days=10), 6 * 24, tmax=10, tmin=3)
    S2 = _series(eng, warm, dvs=0.3)
    st0 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    st, rec2 = eng.run(S2, st, warm["precipitation"], np.zeros(len(S2.times)))
    assert rec2["melt"].sum() > 5.0                                # тане
    assert abs(_closure(eng, st0, st, rec2)) < 1e-6
    # баланс снігу: було 20 мм, стало 20 − Σтанення
    assert st["swe"][0] == pytest.approx(20.0 - rec2["melt"].sum(), abs=1e-9)


def test_frozen_soil_increases_runoff():
    P, _ = _P()
    eng = WaterBalanceEngine(P)
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    wx = make_weather(LAT, LON, start, 2 * 24, tmax=4, tmin=1, rain_events=[(10, 6.0, 5, 90)])
    W = {k: v for k, v in wx.items() if k != "times"}
    d = np.full(len(wx["times"]), 0.3)
    thawed = eng.prepare(wx["times"], W, d, np.full(len(d), 3.0))
    frozen = eng.prepare(wx["times"], W, d, np.full(len(d), -1.0))
    _, r1 = eng.run(thawed, eng.init_state(1, 60.0, zr0=thawed.zr[0]), wx["precipitation"], np.zeros(len(d)))
    _, r2 = eng.run(frozen, eng.init_state(1, 60.0, zr0=frozen.zr[0]), wx["precipitation"], np.zeros(len(d)))
    assert r2["ro"].sum() > r1["ro"].sum() + 1.0


# ---------------------------------------------------------------- 3. два шари
def test_rain_wets_upper_layer_first_and_mass_closes():
    P, _ = _P()
    eng = WaterBalanceEngine(P)
    wx = make_weather(LAT, LON, datetime(2026, 7, 1, tzinfo=timezone.utc), 3 * 24, tmax=26, tmin=15,
                      rain_events=[(30, 4.0, 3, 90)])
    S = _series(eng, wx)
    st = eng.init_state(1, 120.0, zr0=S.zr[0])           # суха коренева зона 1.4 м
    st0 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    st, rec = eng.run(S, st, wx["precipitation"], np.zeros(len(S.times)))
    j = 34
    assert rec["theta_up"][0, j] > rec["theta"][0, j] + 0.02     # верх вологіший за середнє після дощу
    assert abs(_closure(eng, st0, st, rec)) < 1e-6


def test_sensor_layer_operator_reduces_innovation_after_rain():
    """Датчик на 20 см бачить зволожений верхній шар; оператор шару пояснює це без «помилки моделі»."""
    P, _ = _P()
    eng = WaterBalanceEngine(P)
    wx = make_weather(LAT, LON, datetime(2026, 7, 1, tzinfo=timezone.utc), 2 * 24, tmax=26, tmin=15,
                      rain_events=[(10, 4.0, 3, 90)])
    S = _series(eng, wx)
    st, rec = eng.run(S, eng.init_state(1, 120.0, zr0=S.zr[0]), wx["precipitation"], np.zeros(len(S.times)))
    j = 15
    theta_sensor = float(rec["theta_up"][0, j])            # «правда» верхнього шару
    zr, zu = S.zr[j], min(Z_UP, S.zr[j])
    dr, du = float(rec["dr"][0, j]), float(rec["du"][0, j])
    fc = eng.hyd["theta_fc"]
    k1 = SoilMoistureKalman(dr, 10.0, du0=du)
    i1 = k1.update(theta_sensor, zr, fc, 0.01, 0.015, 20, layer=dict(name="upper", zu=zu, tawu=1e9))
    k2 = SoilMoistureKalman(dr, 10.0, du0=du)
    i2 = k2.update(theta_sensor, zr, fc, 0.01, 0.015, 20)
    assert abs(i1["innovation"]) < 0.005 < abs(i2["innovation"])


# ---------------------------------------------------------------- 4. якір FC потребує підтвердження датчика
def test_fc_anchor_requires_sensor_confirmation():
    flux = dict(et=np.array([0.1]), t=np.array([0.05]), pe=np.array([0.0]), inet=np.array([0.0]),
                dp=np.array([0.5]), taw=np.array([200.0]))
    k = SoilMoistureKalman(0.0, 40.0)
    for _ in range(10):
        k.predict(0.0, flux)
    assert k.n_pins == 0                                    # модель «думає», що дренаж, але датчик мовчить
    k.update(0.40, 1.4, 0.408, 0.01, 0.015, 30)             # датчик біля FC → підтвердження
    for _ in range(5):
        k.predict(0.0, flux)
    assert k.n_pins > 0


# ---------------------------------------------------------------- 5. ETo ASCE, CN λ, інфільтрація
def test_asce_hourly_consistent_with_daily():
    wx = make_weather(LAT, LON, datetime(2026, 7, 1, 21, tzinfo=timezone.utc), 24 * 6, tmax=28, tmin=14)
    W = {k: v for k, v in wx.items() if k != "times"}
    res = {}
    for m in ("fao56", "asce"):
        met = meteo.compute_local_eto_series(wx["times"], W, LAT, LON, 200, {}, method=m)
        res[m] = met
    rows = []
    for dday in range(1, 5):
        idx = slice(24 * dday, 24 * dday + 24)
        t = res["asce"]["t"][idx]; rh = res["asce"]["rh"][idx]; u2 = res["asce"]["u2"][idx]
        daily = meteo.eto_daily(t.max(), t.min(), res["asce"]["rs_mj"][idx].sum(), float(u2.mean()), LAT,
                                wx["times"][24 * dday].timetuple().tm_yday, 200, rhmax=rh.max(), rhmin=rh.min())
        rows.append((res["fao56"]["eto_station"][idx].sum(), res["asce"]["eto_station"][idx].sum(), daily))
    r = np.array(rows)
    # обидві погодинні схеми в межах ±5% від добової формули; ASCE вища за FAO-56 (менший денний опір 50 с/м)
    assert abs(r[:, 1].sum() / r[:, 2].sum() - 1) < 0.05 and abs(r[:, 0].sum() / r[:, 2].sum() - 1) < 0.05
    assert r[:, 1].sum() > r[:, 0].sum()


def test_cn_lambda_005_conversion_hawkins():
    # CN0.2 = 50 → S0.2 = 10 дюймів → S0.05 = 1.33·10^1.15 ≈ 18.8 дюйма (Hawkins et al., 2002)
    s02 = (25400 / 50 - 254) / 25.4
    assert 1.33 * s02 ** 1.15 == pytest.approx(18.8, abs=0.1)
    assert float(soil.runoff_cn(20.0, 78.0, 0.05)) > float(soil.runoff_cn(20.0, 78.0, 0.2))


def test_sprinkler_intensity_above_ksat_runs_off():
    P, _ = _P(irrigation_method="center_pivot", application_rate_mm_h=40.0)
    eng = WaterBalanceEngine(P)
    wx = make_weather(LAT, LON, datetime(2026, 7, 1, tzinfo=timezone.utc), 24, tmax=26, tmin=15)
    S = _series(eng, wx)
    irr = np.zeros(len(S.times)); irr[5] = 30.0
    _, rec = eng.run(S, eng.init_state(1, 80.0, zr0=S.zr[0]), np.zeros(len(S.times)), irr)
    lost = rec["ro"].sum()
    expect = 30.0 * eng.method["ea"] * (1 - S.drift[5]) * (1 - eng.hyd["ksat_mm_h"] / 40.0)
    assert lost == pytest.approx(expect, rel=0.01)


# ---------------------------------------------------------------- 6. ліміт очікувань дощу, довідник, калібрування
def test_rain_wait_limit_stops_endless_postponing():
    P, _ = _P()
    eng = WaterBalanceEngine(P)
    wx = make_weather(LAT, LON, datetime(2026, 7, 15, 7, tzinfo=timezone.utc), 72, tmax=27, tmin=15,
                      rain_events=[(10, 10.0, 3, 95)])
    S = _series(eng, wx)
    raw = float(S.p[0] * eng.taw(S.zr[0]))
    st = dict(dr=raw + 3.0, de=eng.rew, x=0.0, fw=1.0, zr=float(S.zr[0]), du=None, swe=0.0)
    pr = wx["precipitation"]; prob = wx["precipitation_probability"]
    d0 = decide(eng, S, dict(st), 5.0, pr, prob, dict(P, _rain_waits=0), rng=np.random.default_rng(1))
    assert d0["action"] == "postpone" and d0["waiting_for_rain"]          # спершу — чекаємо дощ
    d2 = decide(eng, S, dict(st), 5.0, pr, prob, dict(P, _rain_waits=2), rng=np.random.default_rng(1))
    assert not d2["waiting_for_rain"] and any(x["code"] == "rain_wait_limit" for x in d2["reasons"])


def test_handbook_constants_used_for_preset():
    P, prov = _P()
    assert P["theta_fc"] == pytest.approx(0.408) and P["theta_wp"] == pytest.approx(0.235)
    assert prov["theta_fc"]["source"] == "preset" and "Д65" in prov["theta_fc"]["note"]
    P2, prov2 = _P(soil_type_ua="sod_podzolic")            # для нього вимірювань немає → педотрансфер
    assert P2["theta_fc"] is None and prov2["theta_fc"]["source"] == "derived"


def test_sensor_calibration_is_applied(tmp_path):
    from test_hydrobrain import NOW, _brain, _wx
    b, fid = _brain(tmp_path)
    b.update_params(fid, {"sensor_calib": [0.02, 1.0]}, mode="expert")
    b.add_sensor_reading(fid, 0.30, depth_cm=20, time=NOW - timedelta(hours=2))
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["moisture"]["last_sensor"]["theta_obs"] == pytest.approx(0.32, abs=1e-6)
