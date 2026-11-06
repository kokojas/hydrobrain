import sys, tempfile; from pathlib import Path; _R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_R / 'tests')); sys.path.insert(0, str(_R))
from test_hydrobrain import _wx, NOW
from hydrobrain import HydroBrain
from datetime import timedelta
def case(th, rain, label, method="drip", **kw):
    b = HydroBrain(tempfile.mkdtemp(prefix='hbq_'), offline=True)
    inp = dict(lat=49.23, lon=28.47, crop="maize", sowing_date="2026-05-01", irrigation_method=method, soil_type_ua="chornozem_typical", area_m2=10000, flow_rate_m3_h=30); inp.update(kw)
    fid = b.create_field(inp)["field_id"]
    for h in (1, 13, 25, 37):
        b.add_sensor_reading(fid, th, depth_cm=40, time=NOW - timedelta(hours=h))
    r = b.recommend(fid, now=NOW, weather=_wx(rain))
    print("---", label); print(r["headline"]); print("  ", r["summary"][len(r["headline"]):]); print("   details:", r["details"][2:], "| when", r["when_local"], "| next", r["next_check_hours"])
    return r
if __name__ == "__main__":
    case(0.37, [], "вологий, без дощу")
    case(0.345, [], "трохи нижче, без дощу")
    case(0.34, [(10,4,3,90)], "дощ 12 мм через 10 год")
    case(0.33, [(8, 6, 3, 90)], "межа RAW + дощ 18 мм через 8 год")
    case(0.30, [], "сухо")
    case(0.34, [], "дощування", method="sprinkler")
