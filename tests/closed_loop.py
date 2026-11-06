"""Замкнений цикл: «справжнє» поле (Kc на 15% вищий), датчик з шумом, користувач поливає на 30% більше."""
import shutil, sys, tempfile
from datetime import datetime, timedelta, timezone
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from hydrobrain import HydroBrain
from hydrobrain.balance import WaterBalanceEngine
from hydrobrain import crop as cropm
from hydrobrain.synthetic import make_weather

LAT, LON = 47.0, 32.0   # Миколаївщина
BIAS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.015
EVERY_H = int(sys.argv[2]) if len(sys.argv) > 2 else 48
MODE = "expert" if EVERY_H < 48 else "beginner"
start = datetime(2026, 5, 10, 0, tzinfo=timezone.utc)
days = 80
rain_events = [(24*d + 15, 5.0, 2, 75) for d in (6, 15, 27, 44, 58, 71)]
wx = make_weather(LAT, LON, start, (days + 5) * 24, tmax=29, tmin=16, rain_events=rain_events, seed=3)
b = HydroBrain(tempfile.mkdtemp(prefix="hb_cl_"), offline=True)
inp = dict(lat=LAT, lon=LON, crop="tomato", sowing_date="2026-05-12", irrigation_method="drip",
           soil_type_ua="chornozem_southern", area_m2=2000, flow_rate_m3_h=4, sensor_depth_cm=25)
if MODE == "expert":
    inp.update(sigma_sens=0.012, sigma_spat=0.01)
fid = b.create_field(inp, mode=MODE)["field_id"]
P = b.load_field(fid)["params"]
truth = WaterBalanceEngine(P)
times = wx["times"]
dvs = cropm.development_series(times, wx["temperature_2m"], datetime(2026,5,12,tzinfo=timezone.utc), truth.crop)
S = truth.prepare(times, {k: v for k, v in wx.items() if k != "times"}, dvs, wx["soil_temperature_18cm"])
tst = truth.init_state(1, 0.15 * truth.taw(S.zr[0]), zr0=S.zr[0])
irr = np.zeros(len(times)); i_prev = 0
rng = np.random.default_rng(0)
log = []
for d in range(3, days):
    now = start + timedelta(days=d, hours=4)        # 07:00 за Києвом
    i_now = next(i for i, t in enumerate(times) if t >= now)
    if EVERY_H >= 48:
        tst, _ = truth.run(S, tst, wx["precipitation"], irr, learn_k=1.15, i0=i_prev, i1=i_now + 1)
        i_prev = i_now + 1
        th_true = float(truth.theta_from_dr(tst["dr"][0], tst["zr"]))
        th_sens = float(truth.hyd['theta_fc'] - tst['du'][0] / (1000.0 * min(0.3, tst['zr'])))
        if d % 2 == 0:
            b.add_sensor_reading(fid, th_sens + BIAS + rng.normal(0, 0.01), depth_cm=25, time=now)
    else:
        for j in range(i_prev, i_now + 1):
            tst, _ = truth.run(S, tst, wx["precipitation"], irr, learn_k=1.15, i0=j, i1=j + 1)
            if j % EVERY_H == 0:
                th = float(truth.hyd['theta_fc'] - tst['du'][0] / (1000.0 * min(0.3, tst['zr'])))  # датчик бачить свій шар (v1.2)
                b.add_sensor_reading(fid, th + BIAS + rng.normal(0, 0.01), depth_cm=25, time=times[j])
        i_prev = i_now + 1
        th_true = float(truth.theta_from_dr(tst["dr"][0], tst["zr"]))
        th_sens = float(truth.hyd['theta_fc'] - tst['du'][0] / (1000.0 * min(0.3, tst['zr'])))
    r = b.recommend(fid, now=now, weather=wx)
    applied = 0.0
    if r["action"] == "irrigate_now" or (r["action"] == "postpone" and r["amount"]
                                          and r["postpone_hours"] < 24):
        # «відкласти до вікна» теж виконується — у запланований час (v1.2: раніше тест його ігнорував)
        hrs = 0 if r["action"] == "irrigate_now" else int(r["postpone_hours"])
        applied = round(r["amount"]["gross_mm"] * 1.3, 1)
        b.log_irrigation(fid, applied, time=now + timedelta(hours=hrs, minutes=30))
        irr[i_now + 1 + hrs] += applied
    f = b.load_field(fid)
    log.append((now.date(), r["action"], r["amount"]["gross_mm"] if r["amount"] else 0, applied,
                round(th_true, 3), round(r["moisture"]["theta"], 3), r["today"]["k_field"],
                r["amount"]["user_multiplier"] if r["amount"] else None, round(r["moisture"]["sensor_bias"], 3)))
for row in log[::4] + log[-3:]:
    print(row)
err = [abs(x[4] - x[5]) for x in log[10:]]
print("MAE θ (модель+датчик vs правда):", round(float(np.mean(err)), 4))
