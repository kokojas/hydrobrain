"""
Ґрунтовий блок: педотрансферні функції Saxton & Rawls (2006), текстурний клас USDA,
гідрологічна група ґрунту, число кривої стоку CN (NEH-630 гл. 10), шар випаровування
(TEW/REW, FAO-56), коефіцієнт дренажу τ (як в AquaCrop), капілярне підживлення.
"""
from __future__ import annotations

import math

import numpy as np

from .databases import CN_TABLE, CR_MAX_BY_TEXTURE, DRAINAGE_CLASSES, REW_BY_TEXTURE


# ---------------------------------------------------------------------------
# Saxton & Rawls (2006), Soil Sci. Soc. Am. J. 70:1569-1578
# ---------------------------------------------------------------------------
def saxton_rawls(sand_pct, clay_pct, om_pct, bd=None, gravel_vol=0.0):
    """
    Повертає вологість при FC (−33 кПа), WP (−1500 кПа), насиченні (м³/м³), Ksat (мм/год),
    з поправкою на фактичну щільність bd (ущільнення) і об'єм каміння gravel_vol (0..1).
    """
    s = np.clip(sand_pct, 0, 100) / 100.0
    c = np.clip(clay_pct, 0, 100) / 100.0
    om = float(np.clip(om_pct, 0, 8))
    t1500t = -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om - 0.013 * c * om + 0.068 * s * c + 0.031
    t1500 = t1500t + (0.14 * t1500t - 0.02)
    t33t = -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om - 0.027 * c * om + 0.452 * s * c + 0.299
    t33 = t33t + (1.283 * t33t ** 2 - 0.374 * t33t - 0.015)
    ts33t = 0.278 * s + 0.034 * c + 0.022 * om - 0.018 * s * om - 0.027 * c * om - 0.584 * s * c + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    ts = t33 + ts33 - 0.097 * s + 0.043
    rho_n = (1.0 - ts) * 2.65
    df = 1.0
    if bd is not None and bd > 0:
        df = float(np.clip(bd / rho_n, 0.9, 1.3))
    rho_df = rho_n * df
    ts_df = 1.0 - rho_df / 2.65
    t33_df = t33 - 0.2 * (ts - ts_df)
    b = (math.log(1500) - math.log(33)) / (math.log(max(t33_df, 0.02)) - math.log(max(t1500, 0.01)))
    lam = 1.0 / b
    ksat = 1930.0 * max(ts_df - t33_df, 1e-4) ** (3.0 - lam)
    # каміння: зменшує об'єм дрібнозему (AWC) і Ksat (Saxton & Rawls, рівн. 19-22, спрощено)
    g = float(np.clip(gravel_vol, 0.0, 0.8))
    fc = t33_df * (1 - g)
    wp = t1500 * (1 - g)
    sat = ts_df * (1 - g)
    alpha = rho_df / 2.65
    ksat_b = ksat * (1 - g) / (1 - g * (1 - 1.5 * alpha)) if g > 0 else ksat
    return dict(theta_fc=float(fc), theta_wp=float(wp), theta_sat=float(sat),
                ksat_mm_h=float(ksat_b), bd_normal=float(rho_n), density_factor=df)


def texture_class(sand, clay):
    """Текстурний клас USDA за піском і глиною (%)."""
    silt = 100 - sand - clay
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 1.5 * clay >= 15 and silt + 2 * clay < 30:
        return "loamy sand"
    if (7 <= clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (clay < 7 and silt < 50 and silt + 2 * clay >= 30):
        return "sandy loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "silt loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "sandy clay loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "clay loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty clay loam"
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 40 and silt >= 40:
        return "silty clay"
    return "clay"


TEXTURE_UK = {
    "sand": "пісок", "loamy sand": "супісок піщаний", "sandy loam": "супісок",
    "loam": "суглинок середній", "silt loam": "суглинок пилуватий", "silt": "пил",
    "sandy clay loam": "суглинок важкий піщанистий", "clay loam": "суглинок важкий",
    "silty clay loam": "суглинок важкий пилуватий", "sandy clay": "глина піщаниста",
    "silty clay": "глина пилувата", "clay": "глина",
}


def hydrologic_group(ksat_mm_h, depth_to_restrictive_m=2.0, gw_depth_m=None):
    """Гідрологічна група A/B/C/D за Ksat (NEH-630 гл. 7, для глибоких профілів)."""
    k_um_s = ksat_mm_h / 3.6
    if depth_to_restrictive_m < 0.5 or (gw_depth_m is not None and gw_depth_m < 0.6):
        return "D"
    if depth_to_restrictive_m <= 1.0:
        thr = (40.0, 10.0, 1.0)
    else:
        thr = (10.0, 4.0, 0.4)
    if k_um_s > thr[0]:
        return "A"
    if k_um_s > thr[1]:
        return "B"
    if k_um_s > thr[2]:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Стік: метод числа кривої (NEH-630, гл. 10)
# ---------------------------------------------------------------------------
def cn_base(cover: str, hsg: str) -> float:
    idx = "ABCD".index(hsg)
    return float(CN_TABLE.get(cover, CN_TABLE["row"])[idx])


def cn_arc1(cn2):
    """CN для сухих умов (ARC I), Chow et al. (1988)."""
    return 4.2 * cn2 / (10.0 - 0.058 * cn2)


def cn_arc3(cn2):
    """CN для вологих умов (ARC III)."""
    return 23.0 * cn2 / (10.0 + 0.13 * cn2)


def cn_slope_adjust(cn2, slope_frac):
    """Поправка CN на ухил (Williams, 1995; таблиці CN складені для ухилу ~5%)."""
    cn3 = cn_arc3(cn2)
    return (cn3 - cn2) / 3.0 * (1.0 - 2.0 * math.exp(-13.86 * slope_frac)) + cn2


def cn_effective(cn2_adj, wetness):
    """
    Динамічний CN: замість 5-денних опадів умова ARC визначається станом моделі.
    wetness = 1 − Dr/TAW (1 — польова вологоємність, 0 — в'янення). Масиви підтримуються.
    """
    w = np.clip(wetness, 0.0, 1.0)
    c1, c3 = cn_arc1(cn2_adj), cn_arc3(cn2_adj)
    return np.where(w < 0.5, c1 + (cn2_adj - c1) * (w / 0.5), cn2_adj + (c3 - cn2_adj) * ((w - 0.5) / 0.5))


def runoff_cn(p_mm, cn, lam=0.2):
    """
    Q = (P − Ia)² / (P − Ia + S), Ia = λ·S, S = 25400/CN − 254 (мм) — NEH-630, рівн. 10-1, 10-13.
    λ = 0.2 — стандарт NEH-630 (таблиці CN складені для нього).
    λ = 0.05 — за Hawkins et al. (2002); тоді S перераховується: S0.05 = 1.33·S0.2^1.15 (S у дюймах, рівн. 8).
    """
    p = np.asarray(p_mm, dtype=float)
    cn = np.clip(cn, 1.0, 99.9)
    s = 25400.0 / cn - 254.0
    if abs(lam - 0.05) < 1e-9:
        s = 1.33 * (s / 25.4) ** 1.15 * 25.4
    ia = lam * s
    return np.where(p > ia, (p - ia) ** 2 / (p - ia + s), 0.0)


def interception(p_mm, lai, fc):
    """Перехоплення опадів листям (Braden, 1985; a = 0.25 мм на одиницю LAI)."""
    a_lai = 0.25 * np.maximum(lai, 0.0)
    p = np.asarray(p_mm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.where(a_lai > 1e-6, a_lai * (1.0 - 1.0 / (1.0 + np.maximum(fc, 0.05) * p / a_lai)), 0.0)
    return np.minimum(ic, p)


# ---------------------------------------------------------------------------
# Шар випаровування, дренаж, капілярне підживлення
# ---------------------------------------------------------------------------
def tew_rew(theta_fc, theta_wp, tex: str, ze=0.10):
    """TEW = 1000(θFC − 0.5θWP)·Ze (FAO-56, рівн. 73); REW — за текстурою (Табл. 19)."""
    tew = 1000.0 * (theta_fc - 0.5 * theta_wp) * ze
    rew = min(REW_BY_TEXTURE.get(tex, 9.0), 0.9 * tew)
    return float(tew), float(rew)


def drainage_tau(ksat_mm_h, drainage_class="good"):
    """
    Коефіцієнт дренажу τ (частка надлишку понад FC, що стікає за добу):
    τ = 0.0866·Ksat^0.35 (Ksat у мм/добу; Raes et al., AquaCrop), × поправка класу дренажу.
    """
    ks_day = ksat_mm_h * 24.0
    tau = 0.0866 * ks_day ** 0.35
    tau *= DRAINAGE_CLASSES.get(drainage_class, DRAINAGE_CLASSES["good"])["tau_mult"]
    return float(np.clip(tau, 0.02, 1.0))


def capillary_rise_day(gw_depth_m, zr_m, tex: str, rel_depletion):
    """
    Капілярне підживлення з ґрунтових вод, мм/добу (спрощена експоненційна модель).
    0, якщо дзеркало ГВ глибше ніж ~1.5 м під коренями (FAO-56: «≈ 0 при > 1 м»).
    rel_depletion = Dr/TAW — сухіший ґрунт «тягне» більше.
    """
    if gw_depth_m is None:
        return 0.0 * np.asarray(rel_depletion, dtype=float)
    d = max(0.0, gw_depth_m - zr_m)
    if d > 1.5:
        return 0.0 * np.asarray(rel_depletion, dtype=float)
    crmax = CR_MAX_BY_TEXTURE.get(tex, 4.0)
    return crmax * math.exp(-2.3 * d) * np.clip(2.0 * np.asarray(rel_depletion, dtype=float), 0.0, 1.0)


def structure_effects(structure: str):
    """Структура (агрегатність) → множник Ksat і зсув CN."""
    return {"good": (1.3, -2.0), "medium": (1.0, 0.0), "poor": (0.6, +3.0)}.get(structure, (1.0, 0.0))
