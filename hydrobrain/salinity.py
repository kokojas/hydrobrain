"""
Блок солей: динаміка ECe у кореневій зоні, внесок добрив (сольовий індекс),
вимивання дренажем, вплив на транспірацію (FAO-56, рівн. 91-92) і норма промивання (FAO-29).

Формула зі схеми проєкту:
    EC_t = EC_base(Geo, pH) + Fd · kFt · e^(−Fl·t)
реалізована як стан X (надлишок ECe понад базовий рівень):
    X ← X + ΔEC_добрив + ΔEC_поливної_води          (внесення)
    X ← X · exp(−DP / W_fc)                          (вимивання; Fl·t ≡ ΣDP / W_fc)
де W_fc = 1000·θFC·Zr — запас води в кореневій зоні при FC, мм.
"""
from __future__ import annotations

import numpy as np

TDS_PER_EC = 640.0  # мг/л на 1 дСм/м


def ec_base(soil_ece, geo_ece_add, ph):
    """Базова ECe: ґрунт + внесок материнської породи + слабка залежність від лужності."""
    alk = max(0.0, ph - 7.5) * 0.3
    return float(soil_ece + geo_ece_add + alk)


def fert_delta_ece(dose_kg_ha, salt_index, w_fc_mm):
    """
    Приріст ECe від внесення добрива: ΔTDS (мг/л) = доза(кг/га)·SI / W(мм);
    ECsw = TDS/640; ECe ≈ ECsw/2.  kFt = SI/100.
    """
    if dose_kg_ha <= 0 or w_fc_mm <= 0:
        return 0.0
    tds = dose_kg_ha * salt_index / w_fc_mm
    return tds / TDS_PER_EC / 2.0


def irrigation_delta_ece(i_net_mm, ec_iw, w_fc_mm):
    """Сіль, принесена поливною водою (ECiw, дСм/м), у перерахунку на ECe."""
    return np.asarray(i_net_mm) * ec_iw / np.maximum(w_fc_mm, 1.0) / 2.0


def leach(x_excess, dp_mm, w_fc_mm):
    return x_excess * np.exp(-np.asarray(dp_mm) / np.maximum(w_fc_mm, 1.0))


def ks_salinity(ece, ece_thr, b, ky):
    """Ks через засолення без водного стресу (FAO-56, рівн. 91). Обмежено [0.3, 1]."""
    if ky <= 0:
        ky = 1.0
    ks = 1.0 - b / (ky * 100.0) * np.maximum(0.0, np.asarray(ece) - ece_thr)
    return np.clip(ks, 0.3, 1.0)


def leaching_requirement(ec_iw, ece_thr, method="sprinkler"):
    """
    Норма промивання LR (FAO-29, Ayers & Westcot): LR = ECw / (5·ECe_thr − ECw);
    для крапельного — ECw / (2·ECe_max), ECe_max ≈ ECe_thr + 50/b → беремо 2·ECe_thr·2.5.
    """
    if ec_iw <= 0:
        return 0.0
    if method in ("drip", "subsurface_drip"):
        return float(np.clip(ec_iw / (2.0 * 2.5 * ece_thr), 0.0, 0.3))
    den = 5.0 * ece_thr - ec_iw
    return float(np.clip(ec_iw / den if den > 0 else 0.3, 0.0, 0.3))
