"""
Синтетична погода для тестів і демонстрації без інтернету (той самий формат, що й Open-Meteo).
Добовий хід температури/вологості/вітру, радіація = ясне небо × хмарність, задані дощі.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from . import meteo
from .balance import moving_mean as _mm


def make_weather(lat, lon, start_utc: datetime, hours: int, tmax=29.0, tmin=16.0, rh_min=40.0, rh_max=85.0,
                 wind=2.5, cloud=0.15, rain_events=None, rain_prob=None, alt=150.0, seed=1):
    """
    rain_events — список (година_від_старту, мм_за_годину, тривалість_год, ймовірність_%).
    Повертає dict як DataSources.forecast().
    """
    rng = np.random.default_rng(seed)
    times = [start_utc + timedelta(hours=i + 1) for i in range(hours)]
    n = hours
    t = np.zeros(n)
    rh = np.zeros(n)
    u = np.zeros(n)
    sw = np.zeros(n)
    pr = np.zeros(n)
    prob = np.zeros(n)
    day_noise = rng.normal(0, 1.2, n // 24 + 2)
    for i, tt in enumerate(times):
        local_h = (tt.hour + 3) % 24 + 0.5
        phase = math.cos(2 * math.pi * (local_h - 15.0) / 24.0)  # максимум о 15:00
        dn = day_noise[i // 24]
        t[i] = (tmax + tmin) / 2 + (tmax - tmin) / 2 * phase + dn
        rh[i] = (rh_max + rh_min) / 2 - (rh_max - rh_min) / 2 * phase
        u[i] = max(0.3, wind * (1.0 + 0.4 * phase) + rng.normal(0, 0.3))
        ra, _, _ = meteo.ra_hourly(lat, lon, tt)
        rso = (0.75 + 2e-5 * alt) * ra
        sw[i] = rso * (1.0 - 0.75 * cloud) / 0.0036
    for ev in rain_events or []:
        h0, mm_h, dur, p = ev
        for k in range(int(dur)):
            j = int(h0) + k
            if 0 <= j < n:
                pr[j] += mm_h
                prob[j] = max(prob[j], p)
                sw[j] *= 0.25
                rh[j] = max(rh[j], 92.0)
                t[j] -= 3.0
        for k in range(-6, int(dur) + 6):
            j = int(h0) + k
            if 0 <= j < n:
                prob[j] = max(prob[j], p * 0.5)
    if rain_prob is not None:
        prob = np.maximum(prob, rain_prob)
    return dict(times=times, temperature_2m=t, relative_humidity_2m=np.clip(rh, 10, 100), wind_speed_10m=u * 1.34,
                shortwave_radiation=sw, surface_pressure=np.full(n, meteo.pressure_from_altitude(alt) * 10),
                precipitation=pr, precipitation_probability=prob,
                soil_temperature_18cm=_mm(t, 48),
                elevation=alt)
