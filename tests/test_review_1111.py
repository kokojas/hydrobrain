"""Regression cases found during the 1.11 review (all tests use offline sources)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import math

import numpy as np
import pytest

from hydrobrain import HydroBrain
from hydrobrain.decision import infiltration_plan
from hydrobrain.ml_forecast import MLCore, feature_names
from test_hydrobrain import NOW, _wx

INP = dict(lat=49.23, lon=28.47, crop="maize", sowing_date="2026-05-01",
           irrigation_method="drip", soil_type_ua="chornozem_typical")


def field(tmp_path, **extra):
    b = HydroBrain(str(tmp_path), offline=True)
    fid = b.create_field(dict(INP, **extra), mode="expert")["field_id"]
    return b, fid


@pytest.mark.parametrize("extra", [
    {"lat": 91}, {"lon": -181}, {"lat": float("nan")}, {"area_m2": -1},
    {"flow_rate_m3_h": 0}, {"horizon_h": 36.5}, {"horizon_h": 0},
    {"mc_samples": 0}, {"sensor_calib": [0]}, {"sensor_calib": [0, 0]},
    {"forecast_core": "fake"}, {"soil_type_ua": "fake"}, {"geo": "fake"},
    {"sowing_date": "not-a-date"}, {"use_ensemble": "false"},
    {"theta_fc": 0.1, "theta_wp": 0.2}, {"sand": 90, "clay": 40},
    {"preferred_hours": [24]}, {"fertilizers": [{}]},
])
def test_invalid_parameters_are_rejected(tmp_path, extra):
    with pytest.raises(ValueError):
        field(tmp_path, **extra)
    assert not list((tmp_path / "fields").glob("*.json"))


def test_numeric_options_normalized_and_mode_validated(tmp_path):
    b, fid = field(tmp_path, cn_lambda="0.05", horizon_h="36")
    assert b.load_field(fid)["params"]["cn_lambda"] == 0.05
    assert b.load_field(fid)["params"]["horizon_h"] == 36
    with pytest.raises(ValueError):
        b.create_field(INP, mode="typo")
    with pytest.raises(ValueError):
        b.input_schema("typo")


@pytest.mark.parametrize("fid", ["../outside", "/tmp/x", "a/b", "..", "a\\b"])
def test_field_paths_cannot_escape_data_directory(tmp_path, fid):
    b = HydroBrain(str(tmp_path), offline=True)
    with pytest.raises(ValueError):
        b.load_field(fid)


def test_existing_field_cannot_be_overwritten(tmp_path):
    b, fid = field(tmp_path)
    before = b.load_field(fid)
    with pytest.raises(ValueError):
        b.create_field(INP, field_id=fid)
    assert b.load_field(fid) == before


@pytest.mark.parametrize("kind,value", [("sensor", -0.1), ("sensor", 101), ("sensor", float("nan")),
                                        ("irrigation", -2), ("irrigation", 0), ("irrigation", float("inf")),
                                        ("fertilizer", -2)])
def test_invalid_events_do_not_mutate_state(tmp_path, kind, value):
    b, fid = field(tmp_path)
    before = b.load_field(fid)
    method = {"sensor": b.add_sensor_reading, "irrigation": b.log_irrigation,
              "fertilizer": b.add_fertilizer}[kind]
    with pytest.raises(ValueError):
        method(fid, value)
    assert b.load_field(fid) == before


def test_concurrent_sensor_writes_are_not_lost(tmp_path):
    b, fid = field(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: b.add_sensor_reading(fid, 0.3, time=NOW, sensor_id=str(i)), range(40)))
    events = b.load_field(fid)["events"]["sensor"]
    assert len(events) == 40 and len({e["sensor_id"] for e in events}) == 40


def test_arbitrary_horizon_and_repeated_recommendation(tmp_path):
    b, fid = field(tmp_path, horizon_h=36, forecast_core="physics")
    b.add_sensor_reading(fid, 0.33, time=NOW)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    st = b.load_field(fid)["state"]
    r2 = b.recommend(fid, now=NOW, weather=_wx([]))
    assert b.load_field(fid)["state"]["dr"] == st["dr"]
    assert r["risk"] == r2["risk"] and r["action"] == r2["action"]
    assert not r["risk"]["calibrated"]
    json.dumps(r2, allow_nan=False)


def test_future_sensor_does_not_change_present_forecast(tmp_path):
    b1, f1 = field(tmp_path / "a")
    b2, f2 = field(tmp_path / "b")
    for b, f in ((b1, f1), (b2, f2)):
        b.add_sensor_reading(f, 0.3, time=NOW - timedelta(hours=1))
    b2.add_sensor_reading(f2, 0.01, time=NOW + timedelta(days=3))
    r1 = b1.recommend(f1, now=NOW, weather=_wx([]))
    r2 = b2.recommend(f2, now=NOW, weather=_wx([]))
    assert r1["moisture"] == r2["moisture"]
    assert r1["forecast_core"] == r2["forecast_core"]
    assert r1["risk"] == r2["risk"]


def test_late_sensor_replayed_same_hour(tmp_path):
    b, fid = field(tmp_path)
    b.recommend(fid, now=NOW, weather=_wx([]))
    b.add_sensor_reading(fid, 0.3, time=NOW - timedelta(hours=1))
    assert b.load_field(fid)["state"] is None
    b.recommend(fid, now=NOW, weather=_wx([]))
    assert b.load_field(fid)["events"]["sensor"][0]["used"]


def test_small_irrigation_duration_does_not_round_to_zero(tmp_path):
    b, fid = field(tmp_path, area_m2=1, flow_rate_m3_h=0.01, forecast_core="physics")
    b.add_sensor_reading(fid, 0.25, time=NOW)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    am = r["amount"]
    assert am is not None
    assert am["duration_min"] > 0


def test_last_cycle_is_partial_and_total_time_matches_volume():
    p = infiltration_plan(4, 20, 2, 3, 8)
    assert p["cycles"] == 2
    assert p["duration_min"] == pytest.approx(12)
    assert p["last_on_min"] == pytest.approx(2)
    assert p["total_h"] == pytest.approx(1.7)
    assert (p["cycles"] - 1) * p["on_min"] + p["last_on_min"] == pytest.approx(p["duration_min"])


def test_calibration_only_changes_48h_and_stays_in_scope(tmp_path):
    b, fid = field(tmp_path, pstress_calibration=True)
    b.add_sensor_reading(fid, 0.33, time=NOW)
    r = b.recommend(fid, now=NOW, weather=_wx([]))["risk"]
    assert r["calibrated_horizons"] == [48]
    assert r["p_stress_24h"] == r["p_stress_24h_raw"]
    assert r["p_stress_horizon"] == r["p_stress_horizon_raw"]
    b2, f2 = field(tmp_path / "tomato", crop="tomato")
    r2 = b2.recommend(f2, now=NOW, weather=_wx([]))["risk"]
    assert not r2["calibrated"]


def test_all_bundled_models_predict_finite_values():
    core, note = MLCore.load()
    assert core is not None, note
    for target in ("T1", "T2", "T3", "RZ"):
        for h in (1, 2):
            feats = {name: 0.0 for name in feature_names(h)}
            feats["w"] = 0.5
            assert math.isfinite(core.predict(target, h, feats))


def test_http_contract_and_offline_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import server
    monkeypatch.setattr(server, "brain", HydroBrain(str(tmp_path), offline=True))
    monkeypatch.setattr(server, "OFFLINE", True)
    with TestClient(server.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/vendor/leaflet/leaflet.js").status_code == 200
        assert client.get("/schema/typo").status_code == 400
        for route in ("sensor", "irrigation", "fertilizer"):
            body = {"sensor": {"theta": 0.3}, "irrigation": {"amount_mm": 20},
                    "fertilizer": {"dose_kg_ha": 2}}[route]
            assert client.post(f"/fields/missing/{route}", json=body).status_code == 404
        assert client.patch("/fields/missing", json={"crop": "maize"}).status_code == 404
        assert client.post("/fields", json={"inputs": dict(INP, lat=999)}).status_code == 400
        result = client.post("/fields", json={"inputs": INP})
        assert result.status_code == 200
        fid = result.json()["field_id"]
        assert client.post(f"/fields/{fid}/sensor", json={"theta": -1}).status_code == 400
        assert client.post(f"/fields/{fid}/sensor", json={"theta": 30}).status_code == 200
        rec = client.post(f"/fields/{fid}/recommendation")
        assert rec.status_code == 200, rec.text
        assert rec.json()["forecast_core"]["core"] in ("physics", "ml", "ml_local")
        assert client.post(f"/fields/{fid}/irrigation", json={"amount_mm": 10}).status_code == 200
        assert client.post(f"/fields/{fid}/recommendation").status_code == 200


def test_sensor_fault_is_preserved_on_repeat(tmp_path):
    b, fid = field(tmp_path, forecast_core="physics")
    b.add_sensor_reading(fid, 100, time=NOW)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    r2 = b.recommend(fid, now=NOW, weather=_wx([]))
    assert any(x["code"] == "sensor_fault" for x in r["reasons"])
    assert any(x["code"] == "sensor_fault" for x in r2["reasons"])
    assert r["forecast_core"]["state_source"] == r2["forecast_core"]["state_source"]
    assert r["moisture"]["last_sensor"] == r2["moisture"]["last_sensor"]


def test_repeated_rain_wait_does_not_exhaust_limit(tmp_path, monkeypatch):
    import hydrobrain.brain as module
    real_decide = module.decide
    seen = []
    def force_wait(*args, **kwargs):
        P = args[6]
        seen.append(P["_rain_waits"])
        result = real_decide(*args, **kwargs)
        result["waiting_for_rain"] = True
        return result
    monkeypatch.setattr(module, "decide", force_wait)
    b, fid = field(tmp_path, forecast_core="physics")
    for _ in range(4):
        b.recommend(fid, now=NOW, weather=_wx([]))
    assert seen == [0, 0, 0, 0]
    assert b.load_field(fid)["state"]["rain_waits"] == 1


def test_partial_ensemble_does_not_invent_dry_hours():
    from hydrobrain.brain import _align_members
    times = [NOW + timedelta(hours=h) for h in range(10)]
    ens = dict(times=times[:-1], members=np.ones((5, 9)))
    assert _align_members(ens, times) is None


def test_expert_parameters_survive_beginner_edit(tmp_path):
    b, fid = field(tmp_path, theta_fc=0.35, theta_wp=0.15)
    b.update_params(fid, {"area_m2": 200}, mode="beginner")
    assert b.load_field(fid)["params"]["theta_fc"] == 0.35
    assert b.load_field(fid)["params"]["area_m2"] == 200


def test_timezone_override_is_used(tmp_path):
    b, fid = field(tmp_path, tz_offset_h=0)
    r = b.recommend(fid, now=NOW, weather=_wx([]))
    assert r["time_local"] == NOW.strftime("%Y-%m-%d %H:%M")


@pytest.mark.parametrize("broken", ["empty", "missing", "nan", "gap"])
def test_invalid_weather_rejected_without_state_write(tmp_path, broken):
    b, fid = field(tmp_path)
    wx = _wx([])
    if broken == "empty":
        wx["times"] = []
    elif broken == "missing":
        wx.pop("temperature_2m")
    elif broken == "nan":
        wx["temperature_2m"][:] = np.nan
    else:
        wx["times"][3] += timedelta(minutes=30)
    with pytest.raises(ValueError):
        b.recommend(fid, now=NOW, weather=wx)
    assert b.load_field(fid)["state"] is None


def test_late_irrigation_matches_chronological_replay(tmp_path):
    b1, f1 = field(tmp_path / "late", forecast_core="physics")
    b2, f2 = field(tmp_path / "ordered", forecast_core="physics")
    b1.recommend(f1, now=NOW, weather=_wx([]))
    for b, fid in ((b1, f1), (b2, f2)):
        b.log_irrigation(fid, 20, time=NOW - timedelta(hours=2))
    r1 = b1.recommend(f1, now=NOW, weather=_wx([]))
    r2 = b2.recommend(f2, now=NOW, weather=_wx([]))
    assert r1["moisture"]["depletion_mm"] == r2["moisture"]["depletion_mm"]
    assert b1.load_field(f1)["events"]["irrigation"][0]["applied"]


def test_training_csv_has_one_header_under_concurrent_writes(tmp_path):
    import csv
    from hydrobrain.learning import append_training_row
    path = str(tmp_path / "training.csv")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: append_training_row(path, {"a": i}, {"y": 1}, {"id": i}), range(40)))
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40 and len({r["id"] for r in rows}) == 40


@pytest.mark.parametrize("hours", [0.5, 23.75, 24.5])
def test_ml_rollout_uses_exact_elapsed_time(hours):
    from hydrobrain.ml_forecast import rollout
    class ConstantCore:
        def predict(self, *args):
            return 0.24
    wx = _wx([])
    bundle = dict(times=wx["times"], precip=wx["precipitation"],
                  eto=np.zeros(len(wx["times"])), temp=wx["temperature_2m"],
                  rh=wx["relative_humidity_2m"], sw=wx["shortwave_radiation"], wind=wx["wind_speed_10m"])
    now = NOW + timedelta(hours=hours)
    result, hist = rollout(ConstantCore(), "RZ", NOW, 0.5, None, None, now, bundle)
    assert result == pytest.approx(0.5 + 0.24 * hours / 24)
    assert hist[-1][0] == now


def test_location_change_invalidates_cached_climate_and_local_model(tmp_path):
    b, fid = field(tmp_path)
    f = b.load_field(fid)
    f.update(era5l_clim={"RZ": [0.1, 0.3]}, ml_local={"enabled": True}, ml_log=[{"old": True}])
    b._save(f)
    b.update_params(fid, {"lat": 50.1}, mode="expert")
    f = b.load_field(fid)
    assert "era5l_clim" not in f
    assert f["ml_local"] is None and f["ml_log"] == []
