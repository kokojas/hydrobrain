import json, shutil, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrobrain import HydroBrain
from hydrobrain.synthetic import make_weather

def run(future_rain, label, sensor=None, now=datetime(2026,7,15,7,0,tzinfo=timezone.utc), method="drip"):
    b = HydroBrain(tempfile.mkdtemp(prefix="hb_test_"), offline=True)
    f = b.create_field(dict(lat=49.23, lon=28.47, crop="maize", sowing_date="2026-05-01",
                            irrigation_method=method, soil_type_ua="chornozem_typical", area_m2=10000,
                            flow_rate_m3_h=30), mode="beginner")
    start = now - timedelta(days=62)
    past_rain = [(24*d+14, 4.0, 3, 80) for d in (5, 18, 30, 41)]
    hours_past = int((now-start).total_seconds()//3600)
    ev = past_rain + [(hours_past + h, mm, dur, p) for (h, mm, dur, p) in future_rain]
    wx = make_weather(49.23, 28.47, start, hours_past + 96, tmax=30, tmin=17, rain_events=ev)
    if sensor is not None:
        b.add_sensor_reading(f["field_id"], sensor, depth_cm=25, time=now - timedelta(hours=2))
    t0=time.time()
    r = b.recommend(f["field_id"], now=now, weather=wx)
    print(f"=== {label}  ({time.time()-t0:.1f}s)")
    print(r["summary"])
    print("amount:", r["amount"], "| next check h:", r["next_check_hours"])
    print("moisture:", {k: r["moisture"][k] for k in ("theta","theta_sigma","depletion_mm","taw_mm","raw_mm","available_water_pct","root_depth_m")})
    print("today:", r["today"])
    print("risk:", r["risk"])
    print("notes:", r["notes"][:3])
    return b, f, r

if __name__ == "__main__":
    print(json.dumps(HydroBrain(tempfile.mkdtemp(prefix="hb_x_"), offline=True).create_field(dict(lat=49.23, lon=28.47, crop="maize", sowing_date="2026-05-01", irrigation_method="drip", soil_type_ua="chornozem_typical"))["soil_hydraulics"], ensure_ascii=False))
    run([], "A: суха спека")
    run([(20, 6.0, 3, 85)], "B: дощ 18 мм через ~20 год")
    run([], "C: датчик показує вологий ґрунт", sensor=0.36)
    run([], "D: датчик показує сухий ґрунт", sensor=0.20)
