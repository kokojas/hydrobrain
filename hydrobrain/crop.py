"""
Біологічний блок: розвиток культури за температурою (бета-модель Yin et al., 1995),
криві Kcb/Kc (FAO-56, гл. 6-7), кліматична поправка Kc, частка покриття fc,
корені, висота, LAI, густота посіву, здоров'я рослин, провідність продихів.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

KC_MIN = 0.15


# ---------------------------------------------------------------------------
# Бета-модель швидкості розвитку (Yin et al., 1995, рівн. 8c)
# ---------------------------------------------------------------------------
def beta_rate(t_c, tb, to, tc, c=1.0):
    """
    Відносна швидкість розвитку f(T) ∈ [0,1], максимум (=1) при T = To:
    f = [((T−Tb)/(To−Tb)) · ((Tc−T)/(Tc−To))^((Tc−To)/(To−Tb))]^c, поза (Tb,Tc) — 0.
    """
    t = np.asarray(t_c, dtype=float)
    ok = (t > tb) & (t < tc)
    tt = np.where(ok, t, to)
    f = ((tt - tb) / (to - tb)) * ((tc - tt) / (tc - to)) ** ((tc - to) / (to - tb))
    return np.where(ok, np.clip(f, 0.0, 1.0) ** c, 0.0)


def development_series(times_utc, temps_c, sowing: datetime, crop: dict, mode="thermal", dvs0=None, t0=None):
    """
    Стадія розвитку DVS (0 = сходи/сівба … 1 = кінець сезону) для кожної години.
    thermal  — швидкість калібрується так, що при типовій температурі сезону t_typ
               культура проходить усі фази за сумарну кількість днів з FAO-56;
    calendar — DVS = дні від сівби / загальна тривалість (для озимих і за відсутності температур).
    dvs0/t0 — початкова стадія на момент t0 (для продовження вже порахованого ряду).
    """
    total = float(sum(crop["stages"]))
    n = len(times_utc)
    dvs = np.zeros(n)
    if mode == "calendar" or crop.get("winter"):
        for i, tt in enumerate(times_utc):
            dvs[i] = (tt - sowing).total_seconds() / 86400.0 / total
        return np.clip(dvs, 0.0, 1.2)
    tb, to, tc = crop["t_card"]
    f_typ = float(beta_rate(crop["t_typ"], tb, to, tc))
    rate_h = beta_rate(temps_c, tb, to, tc) / (24.0 * total * max(f_typ, 1e-3))
    cur = 0.0 if dvs0 is None else float(dvs0)
    for i, tt in enumerate(times_utc):
        if tt <= sowing:
            dvs[i] = 0.0
            continue
        if t0 is not None and tt <= t0:
            dvs[i] = cur
            continue
        cur += rate_h[i]
        dvs[i] = cur
    return np.clip(dvs, 0.0, 1.2)


def stage_bounds(crop):
    l = np.array(crop["stages"], dtype=float)
    c = np.cumsum(l) / l.sum()
    return c[0], c[1], c[2]  # кінці фаз ini, dev, mid


def stage_name(dvs, crop):
    b1, b2, b3 = stage_bounds(crop)
    if dvs <= 0:
        return "до сходів"
    if dvs < b1:
        return "початкова"
    if dvs < b2:
        return "розвиток"
    if dvs < b3:
        return "середина сезону"
    if dvs <= 1.0:
        return "пізня"
    return "після збирання"


def kcb_curve(dvs, crop, kcb_mid_adj=None, kcb_end_adj=None, key="kcb"):
    """Кусково-лінійна крива Kcb (або Kc при key='kc') від стадії DVS (FAO-56, рис. 25, рівн. 66)."""
    dvs = np.asarray(dvs, dtype=float)
    b1, b2, b3 = stage_bounds(crop)
    ini = crop[f"{key}_ini"]
    mid = crop[f"{key}_mid"] if kcb_mid_adj is None else kcb_mid_adj
    end = crop[f"{key}_end"] if kcb_end_adj is None else kcb_end_adj
    k = np.where(dvs < b1, ini,
                 np.where(dvs < b2, ini + (mid - ini) * (dvs - b1) / (b2 - b1),
                          np.where(dvs < b3, mid,
                                   mid + (end - mid) * np.clip((dvs - b3) / (1.0 - b3), 0, 1))))
    return np.where(dvs <= 0, KC_MIN, np.where(dvs > 1.0, KC_MIN, k))


def climate_adjust(k_tab, u2, rhmin, h):
    """Поправка Kc_mid/Kc_end (> 0.45) на клімат (FAO-56, рівн. 62/65/70)."""
    adj = (0.04 * (u2 - 2.0) - 0.004 * (rhmin - 45.0)) * (h / 3.0) ** 0.3
    return np.where(k_tab > 0.45, k_tab + adj, k_tab)


def kc_max(kcb, u2, rhmin, h):
    """Верхня межа Kc після зволоження (рівн. 72)."""
    a = 1.2 + (0.04 * (u2 - 2.0) - 0.004 * (rhmin - 45.0)) * (h / 3.0) ** 0.3
    return np.maximum(a, kcb + 0.05)


def height_curve(dvs, crop):
    b1, b2, b3 = stage_bounds(crop)
    return crop["h"] * np.clip(0.1 + 0.9 * (np.asarray(dvs) / b2), 0.1, 1.0)


def root_depth(dvs, crop, ds_limit_m):
    """Глибина коренів: від Zr_min до Zr_max до початку середини сезону; обмежена глибиною ґрунту."""
    zmin, zmax = crop["zr"]
    b1, b2, _ = stage_bounds(crop)
    z = zmin + (zmax - zmin) * np.clip(np.asarray(dvs) / b2, 0.0, 1.0)
    return np.minimum(z, max(ds_limit_m, zmin))


def fraction_cover(kcb, kcmax, h):
    """Частка покриття поверхні рослинами fc (рівн. 76)."""
    x = np.clip((kcb - KC_MIN) / np.maximum(kcmax - KC_MIN, 1e-3), 0.0, 1.0)
    return np.clip(x ** (1.0 + 0.5 * h), 0.0, 0.99)


def lai_from_fc(fc):
    """Наближення LAI з fc (обернена формула Беера, k = 0.5)."""
    return -np.log(np.clip(1.0 - fc, 0.01, 1.0)) / 0.5


def kcb_from_lai(lai, kcb_full):
    """Kcb за виміряним LAI (FAO-56, рівн. 97)."""
    return KC_MIN + (kcb_full - KC_MIN) * (1.0 - np.exp(-0.7 * np.asarray(lai)))


def density_coefficient(fc_eff, h):
    """Kd — коефіцієнт густоти/покриття (рівн. 98)."""
    return np.minimum(1.0, np.minimum(1.5 * fc_eff, np.power(np.clip(fc_eff, 0, 1), 1.0 / (1.0 + h))))


def p_adjusted(p_tab, etc_day, dvs, crop, mad_factor=1.0):
    """
    Частка легкодоступної води p: поправка на ETc (FAO-56, прим. до Табл. 22: p = p + 0.04(5 − ETc)),
    зменшення в чутливу фазу (FAO Manual 4, гл. 2) та навчений користувацький множник MAD.
    """
    p = p_tab + 0.04 * (5.0 - np.asarray(etc_day))
    s0, s1 = crop.get("sensitive", (0, 0))
    sens = (np.asarray(dvs) >= s0) & (np.asarray(dvs) <= s1) & (s1 > s0)
    p = np.where(sens, p * 0.85, p)
    return np.clip(p * mad_factor, 0.10, 0.80)


# ---------------------------------------------------------------------------
# Додаткові біологічні/середовищні множники транспірації
# ---------------------------------------------------------------------------
def k_soil_temperature(ts_c, season):
    """Холодний ґрунт гальмує поглинання води коренями (евристика, теплолюбні чутливіші)."""
    if ts_c is None:
        return 1.0
    ts = np.asarray(ts_c, dtype=float)
    lo, hi = (8.0, 15.0) if season == "warm" else (2.0, 8.0)
    return np.clip(0.4 + 0.6 * (ts - lo) / (hi - lo), 0.4, 1.0)


def k_ph(ph, ph_opt):
    """Відхилення pH від оптимуму — доступність води/живлення (евристика: −5% на одиницю pH, мін. 0.8)."""
    lo, hi = ph_opt
    dev = max(0.0, lo - ph, ph - hi)
    return float(max(0.8, 1.0 - 0.05 * dev))


def k_water_temperature(tw_c, season):
    """Холодна поливна вода (шок кореневої системи) — короткочасне зниження поглинання."""
    if tw_c is None:
        return 1.0
    thr = 15.0 if season == "warm" else 8.0
    return float(np.clip(1.0 - 0.02 * max(0.0, thr - tw_c), 0.8, 1.0))


def k_stomatal(gs, gs_ref):
    """Відношення виміряної провідності продихів до еталонної (якщо є порометр)."""
    if gs is None or not gs_ref:
        return 1.0
    return float(np.clip(gs / gs_ref, 0.2, 1.2))
