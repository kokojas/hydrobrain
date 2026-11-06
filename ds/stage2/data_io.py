"""
Читання сирих даних (ds/data/raw) і узгодження масштабів між ERA5-Land і фізичною моделлю.

Ціль — вологість кореневої зони 0–100 см з ERA5-Land, зважена за товщиною шарів (7/21/72 см).
Масштаби різні (у ERA5-Land свої ґрунтові константи), тому все переводиться у ВІДНОСНУ вологість
w = (θ − θ_low) / (θ_high − θ_low), де θ_low/θ_high — 1-й і 99-й перцентилі вегетаційних сезонів
ТРЕНУВАЛЬНИХ років кожної точки (щоб тестові роки не впливали на масштаб).
w ≈ 0 — найсухіший стан за сезони, w ≈ 1 — найвологіший. Фізика працює в тих самих одиницях:
w = 1 − Dr/TAW (тобто θ_high ↔ польова вологоємність, θ_low ↔ в'янення).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ds" / "data" / "raw"
if not RAW.exists():                     # дані можуть лежати в корені проєкту (data/raw)
    RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "ds" / "data" / "processed"

UA_SITES = ["kherson", "mykolaiv", "odesa", "zaporizhzhia", "dnipro", "kropyvnytskyi", "poltava", "kharkiv",
            "cherkasy", "vinnytsia", "kyiv", "zhytomyr", "ternopil", "sumy"]
FR_SITE = "fr_aqui_fraye"
LAYER_W = np.array([7.0, 21.0, 72.0]) / 100.0
SOIL_COLS = ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"]
MET_SOIL_FILE = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "soil_temperature_0_to_7cm"]
WX_COLS = ["wind_speed_10m", "shortwave_radiation", "surface_pressure", "precipitation", "snowfall",
           "et0_fao_evapotranspiration"]
SEASON = ((3, 20), (11, 5))              # погода ERA5 завантажена лише на вегетаційний сезон

TRAIN_YEARS = list(range(2016, 2022))    # 2016–2021
VAL_YEARS = [2022]
TEST_YEARS = [2023, 2024, 2025]          # 2024–2025 — є справжні прогнози погоди
FORECAST_YEARS = [2024, 2025]
FR_SCALE_YEARS = [2013, 2014, 2015]      # Франція не в навчанні: масштаб з перших років станції


def site_meta(name: str) -> dict:
    p = RAW / f"{name}.meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_hourly(name: str) -> pd.DataFrame:
    """Погодинний ряд (UTC, мітка = кінець години): метео + вологість ґрунту, лише сезонні години."""
    s = pd.read_csv(RAW / f"{name}.csv", parse_dates=["time"])
    keep = ["time"] + [c for c in MET_SOIL_FILE + SOIL_COLS if c in s.columns]
    s = s[keep]
    w = pd.read_csv(RAW / f"{name}_wx.csv", parse_dates=["time"])
    df = s.merge(w, on="time", how="inner").sort_values("time").reset_index(drop=True)
    df["theta_rz"] = df[SOIL_COLS].to_numpy() @ LAYER_W
    df["year"] = df["time"].dt.year
    return df


def load_forecast(name: str) -> pd.DataFrame | None:
    """Архів прогнозів: колонки <змінна>_l1 / _l2 (видані за 1 і 2 доби до години)."""
    p = RAW / f"{name}_prev.csv"
    if not p.exists():
        return None
    f = pd.read_csv(p, parse_dates=["time"])
    ren = {c: c.replace("_previous_day1", "_l1").replace("_previous_day2", "_l2") for c in f.columns}
    return f.rename(columns=ren)


def scale_bounds(df: pd.DataFrame, years) -> dict:
    """θ_low/θ_high (1-й/99-й перцентилі) для кореневої зони і кожного шару — на заданих роках."""
    sub = df[df["year"].isin(years)]
    out = {}
    for c in ["theta_rz"] + SOIL_COLS:
        lo, hi = np.nanpercentile(sub[c], 1), np.nanpercentile(sub[c], 99)
        out[c] = (float(lo), float(hi))
    return out


def to_rel(theta, bounds):
    lo, hi = bounds
    return (np.asarray(theta, float) - lo) / max(hi - lo, 1e-6)


# 1.10: профілі 0–30 і 0–55 см — СПОЧАТКУ агрегуємо шари у м³/м³, ПОТІМ нормалізуємо (так само, як у застосунку
# і для датчиків). У версіях до 1.10 під час навчання усереднювались уже нормалізовані шари — інша величина.
PROFILE_MIX = {"T2": (7.0, 21.0, 2.0), "T3": (7.0, 21.0, 27.0)}


def profile_theta(df: pd.DataFrame, target: str) -> pd.Series:
    a, b, c = PROFILE_MIX[target]
    return (a * df[SOIL_COLS[0]] + b * df[SOIL_COLS[1]] + c * df[SOIL_COLS[2]]) / (a + b + c)


def layer_profiles(name: str, years=None) -> pd.DataFrame:
    """Відносна вологість профілів T2/T3 о 00:00 UTC: агрегування в м³/м³ → нормалізація за 1/99 перцентилями
    сезонних годин років навчання (як scale_bounds)."""
    h = load_hourly(name)
    years = years or (FR_SCALE_YEARS if name == FR_SITE else TRAIN_YEARS)
    out = pd.DataFrame(index=h["time"])
    sub = h["year"].isin(years).to_numpy()
    for t in PROFILE_MIX:
        th = profile_theta(h, t).to_numpy()
        lo, hi = np.nanpercentile(th[sub], 1), np.nanpercentile(th[sub], 99)
        out[f"w_{t}"] = to_rel(th, (lo, hi))
    out = out[out.index.hour == 0]
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    return out
