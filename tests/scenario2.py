import shutil, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrobrain import HydroBrain
from hydrobrain.synthetic import make_weather

NOW = datetime(2026, 7, 15, 7, 0, tzinfo=timezone.utc)   # 10:00 за Києвом

def setup(method="drip", crop="maize", extra=None, mode="beginner", ddir=None):
    ddir = ddir or tempfile.mkdtemp(prefix="hb2_")
    b = HydroBrain(ddir, offline=True)
    inp = dict(lat=49.23, lon=28.47, crop=crop, sowing_date="2026-05-01", irrigation_method=method,
               soil_type_ua="chornozem_typical", area_m2=10000, flow_rate_m3_h=30)
    inp.update(extra or {})
    f = b.create_field(inp, mode=mode)
    fid = f["field_id"]
    start = NOW - timedelta(days=60)
    # історія: полив кожні 4 доби по 30 мм з 20-го дня
    t = start + timedelta(days=20, hours=2)
    while t < NOW - timedelta(days=1):
        b.log_irrigation(fid, 30.0, time=t)
        t += timedelta(days=4)
    return b, fid, start

def weather(start, future, now=NOW, wind=2.5, tmax=30):
    hp = int((now - start).total_seconds() // 3600)
    past = [(24*d+14, 4.0, 3, 80) for d in (5, 18, 30, 41)]
    ev = past + [(hp + h, mm, dur, p) for (h, mm, dur, p) in future]
    return make_weather(49.23, 28.47, start, hp + 96, tmax=tmax, tmin=17, rain_events=ev, wind=wind)

def show(label, r):
    print(f"--- {label}\n{r['summary']}\n   amount={r['amount']}  risk48={r['risk']['p_stress_48h']} waste={r['risk']['expected_waste_mm']}  rain={ {k: round(v,1) if isinstance(v,float) else v for k,v in r['rain'].items()} }")

if __name__ == "__main__":
    b, fid, start = setup()
    r = b.recommend(fid, now=NOW, weather=weather(start, []), ensemble=None); show("Сухо, крапельний", r)
    b, fid, start = setup()
    r = b.recommend(fid, now=NOW, weather=weather(start, [(18, 8.0, 3, 90)])); show("Сильний дощ 24 мм через 18 год (90%)", r)
    b, fid, start = setup()
    r = b.recommend(fid, now=NOW, weather=weather(start, [(30, 1.0, 2, 40)])); show("Слабкий дощ 2 мм через 30 год (40%)", r)
    b, fid, start = setup(method="sprinkler")
    r = b.recommend(fid, now=NOW, weather=weather(start, [], wind=6.5)); show("Дощування, вітер", r)
    b, fid, start = setup(crop="potato", extra=dict(mulch_type="organic"))
    r = b.recommend(fid, now=NOW, weather=weather(start, [], tmax=34)); show("Картопля, мульча, спека 34", r)
    print("today:", r["today"])
    print("forecast_days:", r["forecast_days"])
