"""
Командний рядок:
    python -m hydrobrain demo              — демонстрація на синтетичній погоді (без інтернету)
    python -m hydrobrain demo --live       — те саме, але з реальними API (Open-Meteo, DEM, SoilGrids, геологія)
    python -m hydrobrain schema beginner   — поля форми для новачка (JSON)
    python -m hydrobrain recommend <field_id> [--data hydro_data]
    python -m hydrobrain diag --lat 49.3 --lon 28.3   — перевірити кожне джерело даних окремо
Опції demo: --lat/--lon (координати поля), --crop, --sowing (дата сівби РРРР-ММ-ДД)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

from . import HydroBrain
from .synthetic import make_weather


def main():
    ap = argparse.ArgumentParser(prog="hydrobrain")
    ap.add_argument("cmd", choices=["demo", "schema", "recommend", "diag"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--data", default="hydro_data")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--lat", type=float, default=49.30)
    ap.add_argument("--lon", type=float, default=28.30)
    ap.add_argument("--crop", default="tomato")
    ap.add_argument("--sowing", default=None)
    a = ap.parse_args()
    if a.cmd == "diag":
        diag(a)
        return
    if a.cmd == "schema":
        print(json.dumps(HydroBrain.input_schema(a.arg or "beginner"), ensure_ascii=False, indent=1))
        return
    if a.cmd == "recommend":
        b = HydroBrain(a.data)
        print(json.dumps(b.recommend(a.arg), ensure_ascii=False, indent=1))
        return
    # ---- demo
    b = HydroBrain(a.data, offline=not a.live)
    sowing = a.sowing or ((datetime.now(timezone.utc) - timedelta(days=60)).date().isoformat() if a.live
                          else "2026-05-10")
    inputs = dict(lat=a.lat, lon=a.lon, crop=a.crop, sowing_date=sowing, irrigation_method="drip",
                  soil_type_ua="auto" if a.live else "chornozem_typical", area_m2=1000, flow_rate_m3_h=2.5)
    f = b.create_field(inputs, mode="beginner", name="Демо")
    print("Поле:", f["field_id"])
    print("Ґрунт (розраховано):", f["soil_hydraulics"])
    for w in f["warnings"]:
        print("  !", w)
    if a.live:
        r = b.recommend(f["field_id"])
    else:
        now = datetime(2026, 7, 15, 7, 0, tzinfo=timezone.utc)
        b.add_sensor_reading(f["field_id"], 0.31, depth_cm=20, time=now - timedelta(hours=2))
        wx = make_weather(a.lat, a.lon, now - timedelta(days=60), 60 * 24 + 96,
                          rain_events=[(60 * 24 + 20, 5.0, 3, 80)])
        r = b.recommend(f["field_id"], now=now, weather=wx)
    print("\n" + r["summary"])
    print("\nДетально:")
    for d in r["details"]:
        print(" •", d)
    print("\nНорма:", r["amount"])
    print("Вологість:", {k: r["moisture"][k] for k in ("theta", "theta_sigma", "depletion_mm", "taw_mm", "raw_mm")})


def diag(a):
    """Перевіряє кожне джерело окремо й показує, що саме воно повернуло (кеш не використовується)."""
    from .sources import DataSources
    import tempfile
    ds = DataSources(tempfile.mkdtemp(prefix="hb_diag_"))
    checks = [("Прогноз Open-Meteo", lambda: ds.forecast(a.lat, a.lon, 1, 2)),
              ("Ансамбль опадів", lambda: ds.ensemble_precip(a.lat, a.lon, 2)),
              ("Рельєф (DEM)", lambda: ds.terrain(a.lat, a.lon)),
              ("Геологія", lambda: ds.geology(a.lat, a.lon)),
              ("SoilGrids REST", lambda: ds._soilgrids_rest(a.lat, a.lon)),
              ("SoilGrids WCS", lambda: ds._soilgrids_wcs(a.lat, a.lon))]
    for name, fn in checks:
        try:
            r = fn()
            if isinstance(r, dict) and "times" in r:
                r = {"годин": len(r["times"]), "T_зараз": float(r["temperature_2m"][24]),
                     "опади_48г": float(sum(r["precipitation"][24:72]))}
            elif isinstance(r, dict) and "members" in r:
                r = {"членів": int(r["members"].shape[0]), "годин": int(r["members"].shape[1])}
            print(f"✓ {name}: {r}")
        except Exception as e:
            print(f"✗ {name}: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
