"""
Етап 4 (1.11): спільне для калібрування ймовірностей стресу і бектесту поливу.

Дані: 14 точок України, сезони 2024–2025 — для них є СПРАВЖНІЙ архівний прогноз погоди (запас 1 і 2 доби),
тож рішення ухвалюються за прогнозом, а оцінюються за фактичною погодою.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..stage2 import data_io as io

sys.path.insert(0, str(io.ROOT))
from hydrobrain import crop as cropm  # noqa: E402
from hydrobrain import ml_forecast as mlf  # noqa: E402
from hydrobrain import params as pm  # noqa: E402
from hydrobrain.balance import WaterBalanceEngine  # noqa: E402
from hydrobrain.sources import DataSources  # noqa: E402

YEARS = [2024, 2025]
MET = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation", "precipitation"]


def utc(ts):
    return ts.to_pydatetime().replace(tzinfo=timezone.utc)


def resolve_params(lat, lon, alt, crop="maize", sowing="2024-04-25", method="drip", **extra):
    r = pm.Resolver(DataSources(tempfile.mkdtemp(prefix="hb_s4_"), offline=True))
    P, _, _ = r.resolve(dict(lat=lat, lon=lon, crop=crop, sowing_date=sowing, irrigation_method=method,
                             soil_type_ua="chornozem_typical", **extra), "expert")
    P["alt"] = float(alt if alt is not None else P["alt"])
    return P


def truth_params(P, taw_mult, k_mult):
    """«Справжнє поле»: ті самі параметри, але запас води × taw_mult (через θWP) і транспірація × k_mult."""
    eng = WaterBalanceEngine(P)
    fc, wp = eng.hyd["theta_fc"], eng.hyd["theta_wp"]
    Q = dict(P)
    Q["theta_fc"] = float(fc)
    Q["theta_wp"] = float(fc - taw_mult * (fc - wp))
    return Q, float(k_mult)


def load_site_year(name, year):
    """Погодинний ряд сезону: факт (ERA5) + прогноз (_l1/_l2) + ERA5-Land (вологість). Мітка — кінець години."""
    h = io.load_hourly(name)
    fc = io.load_forecast(name)
    hy = h[h["year"] == year].reset_index(drop=True)
    if fc is not None:
        hy = hy.merge(fc, on="time", how="left")
    hy = hy.ffill().bfill()
    hy["time_utc"] = [utc(t) for t in hy["time"]]
    return hy


def weather_dict(df, suffix=""):
    return {c: df[c + suffix].to_numpy(float) if (c + suffix) in df else df[c].to_numpy(float) for c in MET} | \
        dict(surface_pressure=df["surface_pressure"].to_numpy(float) if "surface_pressure" in df else None)


def dvs_series(hy, P):
    eng = WaterBalanceEngine(P)
    sow = datetime.fromisoformat(P["sowing_date"]).replace(tzinfo=timezone.utc)
    return cropm.development_series(list(hy["time_utc"]), hy["temperature_2m"].to_numpy(float), sow, eng.crop,
                                    "thermal")


def forecast_window(hy, i0, H=48):
    """Години i0+1 … i0+H: погода з прогнозу (запас 1 доба для перших 24 год, 2 доби — далі) і факт."""
    sl = hy.iloc[i0 + 1:i0 + 1 + H]
    lead2 = (np.arange(len(sl)) >= 24)
    fw = {}
    for c in MET:
        a1 = sl[c + "_l1"].to_numpy(float) if c + "_l1" in sl else sl[c].to_numpy(float)
        a2 = sl[c + "_l2"].to_numpy(float) if c + "_l2" in sl else sl[c].to_numpy(float)
        fw[c] = np.where(lead2, a2, a1)
    fw["surface_pressure"] = sl["surface_pressure"].to_numpy(float)
    e1 = sl.get("et0_fao_evapotranspiration_l1", sl["et0_fao_evapotranspiration"]).to_numpy(float)
    e2 = sl.get("et0_fao_evapotranspiration_l2", sl["et0_fao_evapotranspiration"]).to_numpy(float)
    eto_f = np.where(lead2, e2, e1)
    actual = {c: sl[c].to_numpy(float) for c in MET} | dict(surface_pressure=sl["surface_pressure"].to_numpy(float))
    return list(sl["time_utc"]), fw, actual, eto_f


def ml_features_bundle(hy, i0, eto_f, fw, back_h=200):
    """Масиви для mlf.daily_weather: минуле — факт, горизонт — прогноз."""
    a = max(0, i0 - back_h)
    past = hy.iloc[a:i0 + 1]
    times = list(past["time_utc"]) + list(hy.iloc[i0 + 1:i0 + 1 + len(eto_f)]["time_utc"])
    cat = lambda pc, fc: np.r_[past[pc].to_numpy(float), fc]
    return dict(times=times, precip=cat("precipitation", fw["precipitation"]),
                eto=cat("et0_fao_evapotranspiration", eto_f), temp=cat("temperature_2m", fw["temperature_2m"]),
                rh=cat("relative_humidity_2m", fw["relative_humidity_2m"]),
                sw=cat("shortwave_radiation", fw["shortwave_radiation"]), wind=cat("wind_speed_10m", fw["wind_speed_10m"]))


def ml_dw(core, target, now, w_now, w24, w72, bundle):
    wd = mlf.daily_weather(bundle["times"], now, bundle["precip"], bundle["eto"], bundle["temp"], bundle["rh"],
                           bundle["sw"], bundle["wind"])
    return {h: core.predict(target, h, mlf.build_features(now, w_now, w24, w72, wd, h)) for h in (1, 2)}
