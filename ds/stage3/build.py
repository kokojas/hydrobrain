"""
Етап 3: перевірка на РЕАЛЬНИХ датчиках вологості (ISMN, мережа FR_Aqui, Франція).

Станції: fraye, grandcal, hillan2 — у межах ~4 км одна від одної, тому для всіх використовується
погода ERA5 точки fraye (клітинка ERA5 ≈ 25 км, ERA5-Land ≈ 9 км). Станцію hillan (лише 2013–2015)
не використовуємо: її замінила hillan2 на тому самому місці.
Увага: це сосновий ліс на піску (HWSD: пісок 87%, глина 4%), клімат Cfb — дуже далеко від
українських полів. Це жорстка перевірка перенесення, а не «рідні» умови моделі.

Цілі (у відносних одиницях кожного датчика, масштаб — 1-й/99-й перцентилі на роках навчання):
  T1 — поверхня: датчик 5 см        ↔ ERA5-Land шар 0–7 см;
  T2 — профіль 0–30 см (зважені за товщиною датчики до ~30 см) ↔ ERA5-Land (7·θ1 + 21·θ2 + 2·θ3)/30;
  T3 — профіль 0–55 см (датчики до ~55 см) ↔ ERA5-Land (7·θ1 + 21·θ2 + 27·θ3)/55 — наближено, бо шар
       ERA5-Land 28–100 см однорідний усередині і ми беремо з нього лише частку.
Розбиття за часом: навчання/калібрування 2013–2017, тест 2018–2022 (до 9 серпня).

Запуск: python -m ds.stage3.build   →  ds/data/processed/stage3_france.csv.gz, stage3_ua.csv.gz
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import sys

from .. import ismn
from ..stage2 import data_io as io
from ..stage2 import physics_runs as pr
from ..stage2.build_dataset import _daily_weather

sys.path.insert(0, str(io.ROOT))
from hydrobrain.soil_calibration import estimate_fc_wp  # noqa: E402

ISMN_DIR = io.ROOT / "ds" / "data" / "ismn"
CAL_INFO: dict = {}
STATIONS = ["fraye", "grandcal", "hillan2"]
TRAIN_Y = [2013, 2014, 2015, 2016, 2017]
TEST_Y = [2018, 2019, 2020, 2021, 2022]
PROFILE_BOTTOM = 0.30
DEEP_BOTTOM = 0.55          # T3: профіль 0–55 см (спільний для всіх трьох станцій)
WX_SITE = io.FR_SITE


# ---------------------------------------------------------------------------- датчики
def station_depths(st):
    out = {}
    for f in ismn.find_files(ISMN_DIR, st):
        d = ismn.read_stm(f)
        out[d.attrs["depth_to"]] = d
    return dict(sorted(out.items()))


def profile_weights(depths, bottom=PROFILE_BOTTOM):
    """Вага датчика = товщина шару між серединами сусідніх глибин, обрізана на [0, bottom]."""
    use = [z for z in depths if z <= bottom + 0.05]
    w = {}
    for k, z in enumerate(use):
        top = 0.0 if k == 0 else (use[k - 1] + z) / 2
        bot = bottom if k == len(use) - 1 else min((z + use[k + 1]) / 2, bottom)
        w[z] = max(bot - top, 0.0)
    s = sum(w.values())
    return {z: v / s for z, v in w.items()}


def sensor_hourly(st):
    """Погодинний ряд (UTC): θ на 5 см і профіль 0–30 см; лише значення з прапорцем G."""
    deps = station_depths(st)
    cols = {}
    for z, d in deps.items():
        g = d[d["flag"].str.startswith("G")].set_index("time")["sm"]
        cols[z] = g[~g.index.duplicated()]
    wide = pd.DataFrame(cols)
    wts = profile_weights(list(deps))
    prof = sum(wide[z] * w for z, w in wts.items())          # NaN, якщо бракує хоч одного датчика
    wts3 = profile_weights(list(deps), DEEP_BOTTOM)
    deep = sum(wide[z] * w for z, w in wts3.items())
    out = pd.DataFrame({"s_surf": wide[min(deps)], "s_prof": prof, "s_deep": deep})
    out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out, dict(T2=wts, T3=wts3)


def at_midnight(h: pd.DataFrame):
    """Значення о 00:00 UTC = середнє з 23:00, 00:00, 01:00 (зменшує шум одного виміру)."""
    x = h.copy()
    x["day"] = (x.index + pd.Timedelta(hours=1)).floor("D")
    sel = x[(x.index.hour == 23) | (x.index.hour == 0) | (x.index.hour == 1)]
    return sel.groupby("day")[["s_surf", "s_prof", "s_deep"]].mean()


# ---------------------------------------------------------------------------- спільні ознаки
def add_weather_feats(rows_df: pd.DataFrame, dw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for d in rows_df.index:
        r = {}
        for k, nd in (("1", 1), ("3", 3), ("7", 7)):
            win = dw.loc[(dw.index >= d - pd.Timedelta(days=nd)) & (dw.index < d)]
            r["P_past" + k] = win["P"].sum() if len(win) == nd else np.nan
            r["ET0_past" + k] = win["ET0"].sum() if len(win) == nd else np.nan
        winT = dw.loc[(dw.index >= d - pd.Timedelta(days=3)) & (dw.index < d)]
        r["T_past3"] = winT["Tmean"].mean() if len(winT) else np.nan
        for j in (1, 2):
            dd = d + pd.Timedelta(days=j - 1)
            for v in ("P", "ET0", "Tmax", "RH", "SW", "U"):
                r[f"{v}_{j}"] = dw.at[dd, v] if dd in dw.index else np.nan
        out.append(r)
    return pd.concat([rows_df, pd.DataFrame(out, index=rows_df.index)], axis=1)


def add_targets(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    """y_{name}_h = w(d+h) − w(d); минулі зміни; лише для послідовних діб."""
    s = df[col]
    for hh in (1, 2):
        df[f"y_{name}_{hh}"] = s.shift(-hh, freq="D").reindex(df.index) - s
    for k in (1, 3):
        df[f"dw_{name}_past{k}"] = s - s.shift(k, freq="D").reindex(df.index)
    return df


# ---------------------------------------------------------------------------- фізика в абсолютних θ
def season_mask(idx):
    m, d = idx.month, idx.day
    return ((m > 3) | ((m == 3) & (d >= 20))) & ((m < 11) | ((m == 11) & (d <= 5)))


def calibrate_station(sh: pd.DataFrame, h: pd.DataFrame, prior: dict, t0=None, t1=None, col="s_prof"):
    """θFC/θWP за профілем (col) у вікні [t0, t1) (лише вегетаційні сезони, де є опади ERA5)."""
    x = sh[col].dropna()
    x = x[season_mask(x.index)]
    if t0 is not None:
        x = x[(x.index >= t0) & (x.index < t1)]
    r = h.set_index("time_naive")["precipitation"]
    return estimate_fc_wp(x.index.tz_localize("UTC"), x.to_numpy(), prior["theta_fc"], prior["theta_wp"],
                          r.index.tz_localize("UTC"), r.to_numpy())


def theta_physics(meta, soil, theta, mid: pd.DataFrame, h: pd.DataFrame, years, zr=PROFILE_BOTTOM, col="s_prof"):
    """
    Фізика HydroBrain у фізичних одиницях, як у робочій системі: стан задається з θ датчика через
    θFC/θWP (Dr = 1000·(θFC − θ)·Zr), прогноз повертається як зміна θ.
    Повертає DataFrame за добами: dth_T2_1/2 (профіль), dth_T1_1/2 (поверхня через шар випаровування).
    """
    eng = pr.make_engine(meta["lat"], meta["lon"], meta["elevation"], soil=soil, zr=zr, theta=theta)
    fc, wp = eng.hyd["theta_fc"], eng.hyd["theta_wp"]
    lo_s = 0.5 * wp                        # нижня межа шару випаровування (TEW = 1000·(θFC − 0.5θWP)·Ze)
    out = []
    for y in years:
        hy = h[h["year"] == y].reset_index(drop=True)
        if len(hy) < 24 * 20:
            continue
        tt = hy["time_naive"]
        ww = mid.reindex(tt.dt.floor("D"))
        ww.index = tt.index
        is0 = (tt.dt.hour == 0).to_numpy()
        w_prof = np.where(is0, (ww[col] - wp) / (fc - wp), np.nan)
        w_surf = np.where(is0, (ww["s_surf"] - lo_s) / (fc - lo_s), np.nan)
        for sc in (0, 1):
            d, _ = pr.run_season(eng, hy, w_prof, w_surf, sc)
            if len(d):
                out.append(d)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out).drop_duplicates("time").set_index("time")
    d.index = d.index.tz_localize(None)
    res = pd.DataFrame(index=d.index)
    for j in (1, 2):
        res[f"dth_P_{j}"] = (wp + d[f"wphys_{j}"] * (fc - wp)) - mid[col].reindex(d.index)
        res[f"dth_T1_{j}"] = (lo_s + d[f"wsphys_{j}"] * (fc - lo_s)) - mid["s_surf"].reindex(d.index)
    res.attrs.update(theta_fc=fc, theta_wp=wp)
    return res


def rel_physics(meta, soil, h, mid, b, zr, col):
    """Фізика у відносних одиницях датчика (масштаб b[col] ↔ WP…FC), як у M1 Етапу 3."""
    eng = pr.make_engine(meta["lat"], meta["lon"], meta["elevation"], soil=soil, zr=zr)
    out = []
    for y in sorted(h["year"].unique()):
        hy = h[h["year"] == y].reset_index(drop=True)
        if len(hy) < 24 * 20:
            continue
        tt = hy["time_naive"]
        ww = mid.reindex(tt.dt.floor("D"))
        ww.index = tt.index
        is0 = (tt.dt.hour == 0).to_numpy()
        w_p = np.where(is0, io.to_rel(ww[col], b[col]), np.nan)
        w_s = np.where(is0, io.to_rel(ww["s_surf"], b["s_surf"]), np.nan)
        for sc in (0, 1):
            d, _ = pr.run_season(eng, hy, w_p, w_s, sc)
            if len(d):
                out.append(d)
    ph = pd.concat(out).drop_duplicates("time").set_index("time")
    ph.index = ph.index.tz_localize(None)
    return ph


# ---------------------------------------------------------------------------- Франція
def hwsd_soil(st):
    p = next((ISMN_DIR / "FR_Aqui" / st).glob("*static_variables.csv"))
    t = pd.read_csv(p, sep=";", encoding="latin-1")
    top = t[(t["depth_from[m]"] == 0.0)]
    get = lambda q: float(top[top["quantity_name"] == q]["value"].iloc[0])
    return dict(sand=get("sand fraction"), clay=get("clay fraction"),
                om=round(min(get("organic carbon") * 1.724, 8.0), 2))     # органіка ≈ 1.724 × органічний вуглець


def build_france(verbose=True):
    meta = io.site_meta(WX_SITE)
    h = io.load_hourly(WX_SITE)
    h["time_naive"] = h["time"].dt.tz_localize(None) if h["time"].dt.tz is not None else h["time"]
    # ERA5-Land у клітинці fraye (для довідкової «моделі» R і для порівняння)
    e = h.set_index("time_naive")[io.SOIL_COLS]
    e_mid = e[e.index.hour == 0]
    rows = []
    for st in STATIONS:
        sh, wts = sensor_hourly(st)
        mid = at_midnight(sh)
        # масштаб датчика — лише роки навчання, лише сезонні доби
        seas = mid[(mid.index.year.isin(TRAIN_Y)) & (mid.index.month.isin(range(4, 11)))]
        b = {c: (float(np.nanpercentile(seas[c], 1)), float(np.nanpercentile(seas[c], 99)))
             for c in ("s_surf", "s_prof", "s_deep")}
        soil = hwsd_soil(st)
        ph = rel_physics(meta, soil, h, mid, b, PROFILE_BOTTOM, "s_prof")
        ph3 = rel_physics(meta, soil, h, mid, b, DEEP_BOTTOM, "s_deep")[["wphys_1", "wphys_2"]] \
            .rename(columns={"wphys_1": "wphys3_1", "wphys_2": "wphys3_2"})
        dw = _daily_weather(h.assign(time=h["time_naive"]))
        # --- фізика у фізичних одиницях: довідник (HWSD) і калібрування θFC/θWP за датчиком 2013–2017
        years = sorted(h["year"].unique())
        prior_eng = pr.make_engine(meta["lat"], meta["lon"], meta["elevation"], soil=soil, zr=PROFILE_BOTTOM)
        prior = dict(theta_fc=prior_eng.hyd["theta_fc"], theta_wp=prior_eng.hyd["theta_wp"])
        cal = calibrate_station(sh, h, prior, pd.Timestamp(f"{TRAIN_Y[0]}-01-01"),
                                pd.Timestamp(f"{TRAIN_Y[-1] + 1}-01-01"))
        th_prior = theta_physics(meta, soil, None, mid, h, years)
        th_cal = theta_physics(meta, soil, dict(theta_fc=cal["theta_fc"], theta_wp=cal["theta_wp"]), mid, h, years)
        cal3 = calibrate_station(sh, h, prior, pd.Timestamp(f"{TRAIN_Y[0]}-01-01"),
                                 pd.Timestamp(f"{TRAIN_Y[-1] + 1}-01-01"), col="s_deep")
        th3_prior = theta_physics(meta, soil, None, mid, h, years, zr=DEEP_BOTTOM, col="s_deep")
        th3_cal = theta_physics(meta, soil, dict(theta_fc=cal3["theta_fc"], theta_wp=cal3["theta_wp"]), mid, h,
                                years, zr=DEEP_BOTTOM, col="s_deep")
        CAL_INFO[st] = dict(prior=prior, calibrated=dict(theta_fc=cal["theta_fc"], theta_wp=cal["theta_wp"]),
                            fc_from_data=cal["fc_from_data"], wp_from_data=cal["wp_from_data"],
                            n_events=cal["n_events"], notes=cal["notes"],
                            calibrated_0_55=dict(theta_fc=cal3["theta_fc"], theta_wp=cal3["theta_wp"],
                                                 n_events=cal3["n_events"]))
        df = pd.DataFrame(index=mid.index)
        df["w_T1"] = io.to_rel(mid["s_surf"], b["s_surf"])
        df["w_T2"] = io.to_rel(mid["s_prof"], b["s_prof"])
        df["w_T3"] = io.to_rel(mid["s_deep"], b["s_deep"])
        # ERA5-Land тієї ж клітинки, нормована на тих самих роках
        e_d = e_mid.copy()
        e_d["e_T1"] = e_d[io.SOIL_COLS[0]]
        e_d["e_T2"] = (7 * e_d[io.SOIL_COLS[0]] + 21 * e_d[io.SOIL_COLS[1]] + 2 * e_d[io.SOIL_COLS[2]]) / 30
        e_d["e_T3"] = (7 * e_d[io.SOIL_COLS[0]] + 21 * e_d[io.SOIL_COLS[1]] + 27 * e_d[io.SOIL_COLS[2]]) / 55
        e_tr = e_d[e_d.index.year.isin(TRAIN_Y)]
        for c in ("e_T1", "e_T2", "e_T3"):
            lo, hi = np.nanpercentile(e_tr[c], 1), np.nanpercentile(e_tr[c], 99)
            df[c] = io.to_rel(e_d[c].reindex(df.index), (lo, hi))
        df = df[df.index.isin(dw.index)]
        for tgt in ("T1", "T2", "T3"):
            df = add_targets(df, f"w_{tgt}", tgt)
            df = add_targets(df, f"e_{tgt}", "E" + tgt)
        df = add_weather_feats(df, dw)
        df = df.join(ph, how="left").join(ph3, how="left")
        for j in (1, 2):
            df[f"phys_T2_{j}"] = df[f"wphys_{j}"] - df["w_T2"]
            df[f"phys_T1_{j}"] = df[f"wsphys_{j}"] - df["w_T1"]
            df[f"phys_T3_{j}"] = df[f"wphys3_{j}"] - df["w_T3"]
            for tg, key, tp, tc, dk in (("T2", "s_prof", th_prior, th_cal, "P"), ("T1", "s_surf", th_prior, th_cal, "T1"),
                                        ("T3", "s_deep", th3_prior, th3_cal, "P")):
                rng = b[key][1] - b[key][0]
                df[f"physthp_{tg}_{j}"] = tp[f"dth_{dk}_{j}"].reindex(df.index) / rng
                df[f"physthc_{tg}_{j}"] = tc[f"dth_{dk}_{j}"].reindex(df.index) / rng
        df["station"] = st
        df["range_T1"] = b["s_surf"][1] - b["s_surf"][0]
        df["range_T2"] = b["s_prof"][1] - b["s_prof"][0]
        df["range_T3"] = b["s_deep"][1] - b["s_deep"][0]
        df["year"] = df.index.year
        df["doy"] = df.index.dayofyear
        df.attrs["weights"] = wts
        rows.append(df.reset_index(names="time"))
        if verbose:
            print(f"  {st}: {df['w_T1'].notna().sum()} діб з датчиком, ґрунт {soil}, ваги 0–30 "
                  f"{ {k: round(v, 2) for k, v in wts['T2'].items()} }, ваги 0–55 "
                  f"{ {k: round(v, 2) for k, v in wts['T3'].items()} }", flush=True)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------- Україна (навчання ML на ERA5-Land)
def build_ua():
    d = pd.read_csv(io.PROCESSED / "daily.csv.gz", parse_dates=["time"])
    d = d[d["site"].isin(io.UA_SITES)].copy()
    d["w_T1"] = d["w__0_7"]
    from ..core.train_core import _with_profiles
    d = _with_profiles(d)                         # 1.10: T2/T3 — агрегування в м³/м³, потім нормалізація
    out = []
    for (s, y), g in d.groupby(["site", "year"]):
        g = g.set_index("time").sort_index()
        for tgt in ("T1", "T2", "T3"):
            g = add_targets(g, f"w_{tgt}", tgt)
        for j in (1, 2):
            for v in ("P", "ET0", "Tmax", "RH", "SW", "U"):
                g[f"{v}_{j}"] = g[f"{v}_{j}_ideal"]
        out.append(g.reset_index())
    u = pd.concat(out, ignore_index=True)
    keep = ["site", "time", "year", "doy", "w_T1", "w_T2", "w_T3"] + [c for c in u.columns if c.startswith(("y_T", "dw_T"))] \
        + ["P_past1", "P_past3", "P_past7", "ET0_past1", "ET0_past3", "ET0_past7", "T_past3"] \
        + [f"{v}_{j}" for j in (1, 2) for v in ("P", "ET0", "Tmax", "RH", "SW", "U")]
    return u[keep]


def main():
    t0 = time.time()
    fr = build_france()
    fr.to_csv(io.PROCESSED / "stage3_france.csv.gz", index=False)
    (io.PROCESSED / "stage3_soil_calibration.json").write_text(json.dumps(CAL_INFO, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")
    for st, c in CAL_INFO.items():
        print(f"  {st}: θFC {c['prior']['theta_fc']:.3f} → {c['calibrated']['theta_fc']:.3f}, "
              f"θWP {c['prior']['theta_wp']:.3f} → {c['calibrated']['theta_wp']:.3f} ({c['n_events']} подій)")
    ua = build_ua()
    ua.to_csv(io.PROCESSED / "stage3_ua.csv.gz", index=False)
    print(f"Готово за {time.time() - t0:.0f} с: Франція {len(fr)} діб, Україна {len(ua)} діб")


if __name__ == "__main__":
    main()
