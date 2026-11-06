"""
Фізична модель HydroBrain (FAO-56) як ПРОГНОСТИЧНИЙ оператор на 1–2 доби.

Як у робочій системі: у момент рішення (00:00 UTC доби d) стан ґрунту відомий (у системі його дає фільтр
Калмана з датчика; тут — ERA5-Land), і фізика прогнозує його на d+1 і d+2 за погодою.
Реалізація: два «сценарії», що перезапускаються від спостереженого стану через добу зі зсувом
(сценарій 0 — у парні доби, сценарій 1 — у непарні). Кожна доба d є стартом рівно одного сценарію,
а його траєкторія дає прогноз і на d+1, і на d+2.

Режими погоди:
  ideal — фактична погода ERA5 («ідеальний прогноз», верхня межа якості);
  fcst  — справжній архівний прогноз Open-Meteo: перша доба після старту — прогноз із запасом 1 доба,
          друга — із запасом 2 доби (є лише за 2024–2025).

Налаштування фізики для порівняння з ERA5-Land: багаторічна трава (lawn, стадія середини сезону),
коренева зона 1.0 м (як шари ERA5-Land 0–100 см), без поливу, чорнозем типовий для всіх точок.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import timedelta

import numpy as np
import pandas as pd

from . import data_io as io

sys.path.insert(0, str(io.ROOT))
from hydrobrain import params as pm  # noqa: E402
from hydrobrain.balance import WaterBalanceEngine  # noqa: E402
from hydrobrain.sources import DataSources  # noqa: E402

ZR = 1.0
FLUX_KEYS = ("et", "e", "t", "pe", "dp", "ro")


def make_engine(lat, lon, alt, soil_type="chornozem_typical", soil=None, zr=ZR, theta=None):
    """soil — необов'язково dict(sand=%, clay=%, om=%) з метаданих станції; інакше тип ґрунту з довідника.
    theta — необов'язково dict(theta_fc=…, theta_wp=…): калібровані константи замість довідникових."""
    r = pm.Resolver(DataSources(tempfile.mkdtemp(prefix="hb_ds_"), offline=True))
    inp = dict(lat=lat, lon=lon, crop="lawn", sowing_date="2016-03-01", irrigation_method="drip")
    if soil:
        inp.update(soil)
    else:
        inp["soil_type_ua"] = soil_type
    P, _, _ = r.resolve(inp, "expert")
    P["alt"] = float(alt if alt is not None else P["alt"])
    if theta:
        P.update({k: float(v) for k, v in theta.items()})
    eng = WaterBalanceEngine(P)
    eng.crop["zr"] = (zr, zr)
    eng.zr_fixed = zr
    return eng


def _weather_dict(h: pd.DataFrame, suffix=""):
    g = lambda c, alt=None: h[c + suffix].to_numpy(float) if (c + suffix) in h else (
        h[alt].to_numpy(float) if alt is not None and alt in h else None)
    wx = dict(temperature_2m=g("temperature_2m"), relative_humidity_2m=g("relative_humidity_2m"),
              wind_speed_10m=g("wind_speed_10m"), shortwave_radiation=g("shortwave_radiation"),
              surface_pressure=h["surface_pressure"].to_numpy(float) if "surface_pressure" in h else None,
              precipitation=g("precipitation"))
    return wx


def run_season(eng, h: pd.DataFrame, w_rz: np.ndarray, w_top: np.ndarray, scenario: int,
               fc: pd.DataFrame | None = None):
    """
    Один прохід фізики по сезону для одного сценарію (0/1). h — погодинний ряд сезону (сортований).
    fc — якщо задано, погода береться з прогнозу (колонки _l1/_l2) відповідно до запасу від старту.
    Повертає список записів для кожного старту (доби d): прогноз w на d+1 і d+2 і потоки за кожну добу.
    """
    times = [t.to_pydatetime().replace(tzinfo=None) for t in h["time"]]
    from datetime import timezone
    times = [t.replace(tzinfo=timezone.utc) for t in times]
    n = len(times)
    is_mid = np.array([t.hour == 0 for t in times])
    mid_idx = np.nonzero(is_mid)[0]
    resets = set(mid_idx[scenario::2].tolist())
    if 0 not in resets:
        resets.add(0)                         # сценарій стартує з першої години сезону

    if fc is None:
        wx = _weather_dict(h)
    else:
        # для кожної години — запас прогнозу відносно останнього старту цього сценарію
        lead = np.ones(n, int)
        last = 0
        for i in range(n):
            if i in resets:
                last = i
            lead[i] = 1 if (i - last) <= 24 else 2
        hf = h.copy()
        for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation",
                  "precipitation"]:
            v1, v2 = h[c + "_l1"].to_numpy(float), h[c + "_l2"].to_numpy(float)
            hf[c] = np.where(lead == 1, v1, v2)
        wx = _weather_dict(hf)
    dvs = np.full(n, 0.5)
    S = eng.prepare(times, wx, dvs, None, None, tz_offset_h=0.0)
    zr = getattr(eng, "zr_fixed", ZR)
    taw = float(eng.taw(zr))
    sat_cap = eng.sat_unit * zr
    w_start = w_rz[0] if np.isfinite(w_rz[0]) else 0.5
    state = eng.init_state(1, float(np.clip((1 - w_start) * taw, -sat_cap, taw)), zr0=zr)
    rain = np.nan_to_num(wx["precipitation"])
    recs, cur = [], None
    acc = {k: 0.0 for k in FLUX_KEYS}

    def cb(i, st, flux):
        nonlocal cur
        for k in FLUX_KEYS:
            acc[k] += float(np.ravel(flux[k])[0])
        if not is_mid[i] and i != 0:
            return
        w_now = 1.0 - float(st["dr"][0]) / taw
        ws_now = 1.0 - float(st["de"][0]) / eng.tew          # поверхневий шар (~10 см), відносна вологість
        if cur is not None:
            day = len(cur["days"])
            cur["days"].append(dict(w=w_now, ws=ws_now, **{k: acc[k] for k in FLUX_KEYS}))
            if day == 1 or i == n - 1:
                recs.append(cur)
                cur = None
        for k in FLUX_KEYS:
            acc[k] = 0.0
        if i in resets:
            if cur is not None:              # незавершений (сезон скінчився) — відкидаємо
                cur = None
            if not (np.isfinite(w_rz[i]) and np.isfinite(w_top[i])):
                return                       # немає спостереження (пропуск датчика) — цей старт пропускаємо
            st["dr"][0] = float(np.clip((1 - w_rz[i]) * taw, -sat_cap, taw))
            if "du" in st:                   # дво-шарова фізика (1.2+): верхній шар — та сама відносна вологість
                st["du"][0] = max(float(st["dr"][0]), 0.0) * min(0.3, zr) / max(zr, 1e-6)
            st["de"][0] = float(np.clip((1 - w_top[i]) * eng.tew, 0, eng.tew))
            cur = dict(i0=i, time=times[i], w0=1.0 - float(st["dr"][0]) / taw,
                       ws0=1.0 - float(st["de"][0]) / eng.tew, days=[])

    # перша година: стан уже ініціалізовано; викликаємо скидання вручну для узгодженості
    state, _ = eng.run(S, state, rain, np.zeros(n), learn_k=1.0, callback=cb, record=False)
    out = []
    for r in recs:
        if len(r["days"]) < 1:
            continue
        row = dict(time=r["time"], w0_phys=r["w0"], ws0_phys=r["ws0"])
        for j, d in enumerate(r["days"][:2], start=1):
            row[f"wphys_{j}"] = d["w"]
            row[f"wsphys_{j}"] = d["ws"]
            for k in FLUX_KEYS:
                row[f"{k}_{j}"] = d[k]
        out.append(row)
    return pd.DataFrame(out), dict(taw=taw, tew=eng.tew, sat_cap=sat_cap)
