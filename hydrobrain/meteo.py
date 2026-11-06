"""
Метеорологічний блок: еталонна евапотранспірація ETo (FAO-56 Penman-Monteith),
геометрія Сонця, радіація на схилі, мікрокліматичні поправки (затінення,
вентиляція, ухил/експозиція, альбедо) та вплив CO2.

Усі функції працюють і зі скалярами, і з масивами numpy.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

GSC = 0.0820            # сонячна стала, МДж м-2 хв-1
SIGMA_DAY = 4.903e-9    # Стефан-Больцман, МДж K-4 м-2 доба-1
SIGMA_HOUR = 2.043e-10  # Стефан-Больцман, МДж K-4 м-2 год-1


# ---------------------------------------------------------------------------
# Базові психрометричні величини (FAO-56, гл. 3)
# ---------------------------------------------------------------------------
def pressure_from_altitude(z_m):
    """Атмосферний тиск, кПа (FAO-56, рівн. 7)."""
    return 101.3 * ((293.0 - 0.0065 * np.asarray(z_m, dtype=float)) / 293.0) ** 5.26


def psychrometric_constant(p_kpa):
    """γ, кПа/°C (рівн. 8)."""
    return 0.665e-3 * np.asarray(p_kpa, dtype=float)


def sat_vp(t_c):
    """Тиск насиченої пари e°(T), кПа (рівн. 11)."""
    t = np.asarray(t_c, dtype=float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def slope_vp(t_c):
    """Нахил кривої тиску насиченої пари Δ, кПа/°C (рівн. 13)."""
    t = np.asarray(t_c, dtype=float)
    return 4098.0 * sat_vp(t) / (t + 237.3) ** 2


def wind_2m(u_z, z=10.0):
    """Перерахунок швидкості вітру з висоти z на 2 м (рівн. 47)."""
    return np.asarray(u_z, dtype=float) * 4.87 / math.log(67.8 * z - 5.42)


# ---------------------------------------------------------------------------
# Геометрія Сонця
# ---------------------------------------------------------------------------
def _doy(dt: datetime) -> int:
    return dt.timetuple().tm_yday


def inv_rel_distance(j):
    """dr — обернена відносна відстань Земля-Сонце (рівн. 23)."""
    return 1.0 + 0.033 * np.cos(2.0 * np.pi * np.asarray(j) / 365.0)


def declination(j):
    """δ — схилення Сонця, рад (рівн. 24)."""
    return 0.409 * np.sin(2.0 * np.pi * np.asarray(j) / 365.0 - 1.39)


def seasonal_correction(j):
    """Sc — поправка на сезонність сонячного часу, год (рівн. 32-33)."""
    b = 2.0 * np.pi * (np.asarray(j) - 81.0) / 364.0
    return 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)


def ra_daily(lat_deg, j):
    """Позаатмосферна радіація за добу, МДж м-2 доба-1 (рівн. 21)."""
    phi = math.radians(lat_deg)
    dr = inv_rel_distance(j)
    d = declination(j)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(d), -1.0, 1.0))
    return (24 * 60 / np.pi) * GSC * dr * (ws * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(ws))


def solar_hour_angle_utc(t_utc_mid: datetime, lon_deg: float) -> float:
    """Годинний кут ω (рад) для моменту t (UTC) і довготи (східна — додатна)."""
    j = _doy(t_utc_mid)
    hours = t_utc_mid.hour + t_utc_mid.minute / 60.0 + t_utc_mid.second / 3600.0
    solar_time = hours + lon_deg / 15.0 + float(seasonal_correction(j))
    return math.pi / 12.0 * (solar_time - 12.0)


def ra_hourly(lat_deg, lon_deg, t_end_utc: datetime, period_h=1.0):
    """Позаатмосферна радіація за період, що закінчується в t_end (рівн. 28-31), МДж м-2 період-1."""
    j = _doy(t_end_utc)
    phi = math.radians(lat_deg)
    d = float(declination(j))
    dr = float(inv_rel_distance(j))
    mid = t_end_utc.timestamp() - period_h * 1800.0
    w = solar_hour_angle_utc(datetime.fromtimestamp(mid, tz=timezone.utc), lon_deg)
    ws = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(d))))
    w1 = max(-ws, min(ws, w - math.pi * period_h / 24.0))
    w2 = max(-ws, min(ws, w + math.pi * period_h / 24.0))
    if w2 <= w1:
        return 0.0, w, ws
    ra = (12 * 60 / math.pi) * GSC * dr * ((w2 - w1) * math.sin(phi) * math.sin(d)
                                           + math.cos(phi) * math.cos(d) * (math.sin(w2) - math.sin(w1)))
    return max(ra, 0.0), w, ws


def sun_position(lat_deg, lon_deg, t_utc: datetime):
    """Косинус зенітного кута та азимут Сонця (рад, від півночі за годинниковою стрілкою)."""
    j = _doy(t_utc)
    phi = math.radians(lat_deg)
    d = float(declination(j))
    w = solar_hour_angle_utc(t_utc, lon_deg)
    cosz = math.sin(phi) * math.sin(d) + math.cos(phi) * math.cos(d) * math.cos(w)
    cosz = max(-1.0, min(1.0, cosz))
    sinz = math.sqrt(max(1e-12, 1.0 - cosz ** 2))
    cos_az = (math.sin(d) - cosz * math.sin(phi)) / (sinz * math.cos(phi) + 1e-12)
    az = math.acos(max(-1.0, min(1.0, cos_az)))
    if w > 0:  # по обіді Сонце на заході
        az = 2 * math.pi - az
    return cosz, az


# ---------------------------------------------------------------------------
# Радіація на схилі + затінення (мікроклімат)
# ---------------------------------------------------------------------------
def diffuse_fraction_erbs(kt):
    """Частка розсіяної радіації за кліматологічним індексом ясності kt (модель Erbs et al., 1982)."""
    kt = np.clip(np.asarray(kt, dtype=float), 0.0, 1.0)
    return np.where(kt <= 0.22, 1.0 - 0.09 * kt,
                    np.where(kt <= 0.80,
                             0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                             0.165))


def slope_radiation_factor(lat_deg, lon_deg, t_end_utc: datetime, rs_h, ra_h,
                           slope_deg=0.0, aspect_deg=180.0, shade=0.0, ground_albedo=0.2):
    """
    Множник K_rad = Rs_локальна / Rs_горизонтальна для однієї години.
    Пряма радіація перераховується на нахилену площину (кут падіння),
    розсіяна — з коефіцієнтом видимості неба (1+cosβ)/2, плюс відбита від землі.
    shade — частка прямого сонця, що перекрита (дерева, будівлі, сітка), 0..1.
    """
    if rs_h <= 0 or ra_h <= 0:
        return 1.0
    mid = datetime.fromtimestamp(t_end_utc.timestamp() - 1800.0, tz=timezone.utc)
    cosz, az = sun_position(lat_deg, lon_deg, mid)
    if cosz <= 0.03:
        return 1.0 - 0.3 * shade
    kt = min(1.0, rs_h / ra_h)
    kd = float(diffuse_fraction_erbs(kt))
    rb = rs_h * (1.0 - kd)
    rd = rs_h * kd
    beta = math.radians(slope_deg)
    gam = math.radians(aspect_deg)
    sinz = math.sqrt(max(0.0, 1.0 - cosz ** 2))
    cos_inc = cosz * math.cos(beta) + sinz * math.sin(beta) * math.cos(az - gam)
    beam_ratio = max(0.0, cos_inc) / max(cosz, 0.05)
    beam_ratio = min(beam_ratio, 4.0)
    rs_slope = (rb * beam_ratio * (1.0 - shade)
                + rd * (1.0 + math.cos(beta)) / 2.0 * (1.0 - 0.3 * shade)
                + rs_h * ground_albedo * (1.0 - math.cos(beta)) / 2.0)
    return rs_slope / rs_h


# ---------------------------------------------------------------------------
# ETo — FAO-56 Penman-Monteith
# ---------------------------------------------------------------------------
def co2_surface_resistance(co2_ppm=None, rs_ref=70.0):
    """
    Поверхневий опір з урахуванням CO2 (Yang et al., 2019, Nature Climate Change):
    rs = 70·[1 + 0.00034·(CO2 − 300)]. За замовчуванням (None) — стандартні 70 с/м.
    """
    if co2_ppm is None:
        return rs_ref
    return rs_ref * (1.0 + 0.00034 * (co2_ppm - 300.0))


def eto_hourly(t_c, rh, u2, rs_mj, p_kpa, ra_mj, z_m, is_day, rs_rso_prev=0.6,
               albedo=0.23, rs_s_m=70.0, method="fao56"):
    """
    Погодинна ETo, мм/год (FAO-56, рівн. 53).
    rs_mj — сумарна радіація за годину, МДж/м²; ra_mj — позаатмосферна за годину.
    rs_rso_prev — відношення Rs/Rso, яке використовується вночі (останнє денне).
    rs_s_m — поверхневий опір (70 с/м = еталонна трава; змінюється з CO2).
    method — "fao56": постійний опір 70 с/м (Cd = 0.34) і G = 0.1/0.5·Rn удень/вночі (FAO-56, рівн. 53);
             "asce":  стандартизований погодинний варіант ASCE-EWRI (2005), Allen et al. (2006):
                      rs = 50 с/м удень і 200 с/м уночі → Cd = 0.24 / 0.96, G = 0.1/0.5·Rn.
             ASCE-варіант краще узгоджує суму погодинних значень із добовою формулою.
    Повертає (ETo, Rs/Rso, Rn).
    """
    t = float(t_c)
    es = float(sat_vp(t))
    ea = es * max(0.0, min(100.0, rh)) / 100.0
    delta = float(slope_vp(t))
    gamma = float(psychrometric_constant(p_kpa))
    rso = (0.75 + 2e-5 * z_m) * ra_mj
    if is_day and rso > 0.01:
        ratio = max(0.25, min(1.0, rs_mj / rso))
    else:
        ratio = rs_rso_prev
    rns = (1.0 - albedo) * rs_mj
    rnl = SIGMA_HOUR * (t + 273.16) ** 4 * (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * ratio - 0.35)
    rn = rns - rnl
    g = 0.1 * rn if is_day else 0.5 * rn
    co2_k = rs_s_m / 70.0                    # відносна зміна опору через CO2 (1.0, якщо поправку вимкнено)
    if method == "asce":
        cd = (0.24 if is_day else 0.96) * co2_k
    else:
        cd = 0.34 * co2_k
    num = 0.408 * delta * (rn - g) + gamma * (37.0 / (t + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + cd * u2)
    return max(0.0, num / den), ratio, rn


def eto_daily(tmax, tmin, rs_mj, u2, lat_deg, j, z_m, rhmax=None, rhmin=None, rhmean=None,
              tdew=None, p_kpa=None, albedo=0.23, rs_s_m=70.0):
    """
    Добова ETo, мм/добу (FAO-56, рівн. 6). ea — за пріоритетом: RHmax/RHmin, Tdew, RHmean, Tmin.
    """
    tmean = (tmax + tmin) / 2.0
    p = pressure_from_altitude(z_m) if p_kpa is None else p_kpa
    gamma = psychrometric_constant(p)
    es = (sat_vp(tmax) + sat_vp(tmin)) / 2.0
    if rhmax is not None and rhmin is not None:
        ea = (sat_vp(tmin) * rhmax / 100.0 + sat_vp(tmax) * rhmin / 100.0) / 2.0
    elif tdew is not None:
        ea = sat_vp(tdew)
    elif rhmean is not None:
        ea = rhmean / 100.0 * es
    else:
        ea = sat_vp(tmin)
    delta = slope_vp(tmean)
    ra = ra_daily(lat_deg, j)
    rso = (0.75 + 2e-5 * z_m) * ra
    rns = (1 - albedo) * rs_mj
    ratio = np.clip(rs_mj / rso, 0.25, 1.0)
    rnl = SIGMA_DAY * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0) \
        * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * ratio - 0.35)
    rn = rns - rnl
    cd = 0.34 * rs_s_m / 70.0
    num = 0.408 * delta * rn + gamma * 900.0 / (tmean + 273.0) * u2 * (es - ea)
    den = delta + gamma * (1 + cd * u2)
    return float(np.maximum(0.0, num / den))


def eto_hargreaves(tmax, tmin, lat_deg, j):
    """Запасний метод Hargreaves (FAO-56, рівн. 52), мм/добу — коли немає вологості/вітру/радіації."""
    ra = ra_daily(lat_deg, j)
    tmean = (tmax + tmin) / 2.0
    return float(0.0023 * (tmean + 17.8) * math.sqrt(max(0.0, tmax - tmin)) * 0.408 * ra)


# ---------------------------------------------------------------------------
# Повний погодинний розрахунок ETo для ряду метеоданих з мікрокліматом
# ---------------------------------------------------------------------------
VENT_MULT = {"sheltered": 0.6, "normal": 1.0, "exposed": 1.25}


def compute_local_eto_series(times_utc, wx: dict, lat, lon, z_m, micro: dict, method="asce"):
    """
    times_utc — список datetime (UTC), кінець кожної години.
    wx — словник масивів: temperature_2m (°C), relative_humidity_2m (%),
         wind_speed_10m (м/с), shortwave_radiation (Вт/м², середнє за попередню годину),
         surface_pressure (гПа, необов'язково).
    micro — мікрокліматичні параметри: slope_deg, aspect_deg, shade, vent (рядок або число),
            albedo, co2_ppm.
    Повертає словник масивів: eto_station (без мікроклімату), eto_local, k_micro, k_rad,
    u2_local, rs_mj_local, is_day, rh, t, vpd.
    """
    n = len(times_utc)
    t = np.asarray(wx["temperature_2m"], dtype=float)
    rh = np.asarray(wx["relative_humidity_2m"], dtype=float)
    u10 = np.asarray(wx["wind_speed_10m"], dtype=float)
    sw = np.asarray(wx["shortwave_radiation"], dtype=float)
    if wx.get("surface_pressure") is not None:
        p = np.asarray(wx["surface_pressure"], dtype=float) / 10.0
        p = np.where(np.isfinite(p) & (p > 50), p, float(pressure_from_altitude(z_m)))
    else:
        p = np.full(n, float(pressure_from_altitude(z_m)))
    vent = micro.get("vent", "normal")
    vent_mult = VENT_MULT.get(vent, vent if isinstance(vent, (int, float)) else 1.0)
    u2 = wind_2m(u10)
    u2_loc = u2 * vent_mult
    rs = sw * 0.0036  # Вт/м² (середнє за год) → МДж/м²/год
    rs_s = co2_surface_resistance(micro.get("co2_ppm"))
    albedo = micro.get("albedo", 0.23)
    out = {k: np.zeros(n) for k in ("eto_station", "eto_local", "k_rad", "rs_mj_local", "ra")}
    is_day = np.zeros(n, dtype=bool)
    ratio_st, ratio_loc = 0.6, 0.6
    for i, tt in enumerate(times_utc):
        ra, w, ws = ra_hourly(lat, lon, tt)
        day = ra > 0.02
        is_day[i] = day
        k_rad = slope_radiation_factor(lat, lon, tt, rs[i], ra, micro.get("slope_deg", 0.0),
                                       micro.get("aspect_deg", 180.0), micro.get("shade", 0.0)) if day else 1.0
        rs_loc = rs[i] * k_rad
        e_st, r1, _ = eto_hourly(t[i], rh[i], u2[i], rs[i], p[i], ra, z_m, day, ratio_st, method=method)
        e_lc, r2, _ = eto_hourly(t[i], rh[i], u2_loc[i], rs_loc, p[i], ra, z_m, day, ratio_loc,
                                 albedo=albedo, rs_s_m=rs_s, method=method)
        if day and ra > 0.3:  # запам'ятовуємо Rs/Rso при достатньо високому Сонці (для ночі)
            ratio_st, ratio_loc = r1, r2
        out["eto_station"][i] = e_st
        out["eto_local"][i] = e_lc
        out["k_rad"][i] = k_rad
        out["rs_mj_local"][i] = rs_loc
        out["ra"][i] = ra
    with np.errstate(divide="ignore", invalid="ignore"):
        kmic = np.where(out["eto_station"] > 1e-4, out["eto_local"] / out["eto_station"], 1.0)
    out.update(dict(k_micro=kmic, u2=u2, u2_local=u2_loc, is_day=is_day, t=t, rh=rh,
                    vpd=sat_vp(t) * (1 - rh / 100.0), rs_mj=rs, p_kpa=p))
    return out
