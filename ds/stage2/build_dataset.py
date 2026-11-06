"""
Збирає добову таблицю для експерименту: для кожної точки і кожної доби d (00:00 UTC) —
стан ERA5-Land, погода в минулому, погода на d+1/d+2 (факт і, де є, прогноз), прогноз фізики і цілі.

Запуск:  python -m ds.stage2.build_dataset         (≈ 7–10 хв на 1 ядрі; результат кешується)
Результат: ds/data/processed/daily.csv.gz, bounds.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import data_io as io
from . import physics_runs as pr


def _daily_weather(h: pd.DataFrame, suffix=""):
    """Добові агрегати погоди для інтервалу (d 00:00, d+1 00:00] — мітка доби = d (початок)."""
    x = h.copy()
    x["day"] = (x["time"] - pd.Timedelta(seconds=1)).dt.floor("D")   # година, що закінчується о 00:00 → попередня доба
    agg = dict(P=("precipitation" + suffix, "sum"), ET0=("et0_fao_evapotranspiration" + suffix, "sum"),
               Tmean=("temperature_2m" + suffix, "mean"), Tmax=("temperature_2m" + suffix, "max"),
               RH=("relative_humidity_2m" + suffix, "mean"), SW=("shortwave_radiation" + suffix, "mean"),
               U=("wind_speed_10m" + suffix, "mean"))
    agg = {k: v for k, v in agg.items() if v[0] in x.columns}
    return x.groupby("day").agg(**agg)


def build_site(name: str, verbose=True):
    meta = io.site_meta(name)
    h = io.load_hourly(name)
    years = sorted(h["year"].unique())
    scale_years = io.FR_SCALE_YEARS if name == io.FR_SITE else io.TRAIN_YEARS
    b = io.scale_bounds(h, scale_years)
    h["w_rz"] = io.to_rel(h["theta_rz"], b["theta_rz"])
    for c in io.SOIL_COLS:
        h["w_" + c[-9:].replace("to_", "").replace("cm", "")] = io.to_rel(h[c], b[c])
    wcols = [c for c in h.columns if c.startswith("w_")]
    fc = io.load_forecast(name)
    eng = pr.make_engine(meta.get("lat"), meta.get("lon"), meta.get("elevation"))
    rows = []
    for y in years:
        hy = h[h["year"] == y].reset_index(drop=True)
        if len(hy) < 24 * 20:
            continue
        w_rz = hy["w_rz"].to_numpy()
        w_top = hy[wcols[1]].to_numpy()          # шар 0–7 см
        # --- стан ERA5-Land о 00:00 кожної доби
        mid = hy[hy["time"].dt.hour == 0].set_index("time")
        st = mid[wcols + ["theta_rz"]].copy()
        st.index = st.index.tz_localize(None)
        # --- погода: факт (по добах) і минуле
        dw = _daily_weather(hy)
        # --- фізика, режим ideal
        ph = []
        for sc in (0, 1):
            d, info = pr.run_season(eng, hy, w_rz, w_top, sc)
            ph.append(d)
        ph = pd.concat(ph).drop_duplicates("time").set_index("time")
        ph.index = ph.index.tz_localize(None)
        ph = ph.add_suffix("_ideal")
        # --- фізика, режим fcst (лише роки з архівом прогнозів)
        phf = None
        fy = None
        if fc is not None and y in set(fc["time"].dt.year):
            hf = hy.merge(fc, on="time", how="left")
            if hf.filter(like="_l1").notna().all(axis=1).mean() > 0.95:
                hf = hf.ffill()
                phf = []
                for sc in (0, 1):
                    d, _ = pr.run_season(eng, hf, w_rz, w_top, sc, fc=fc)
                    phf.append(d)
                phf = pd.concat(phf).drop_duplicates("time").set_index("time")
                phf.index = phf.index.tz_localize(None)
                phf = phf.add_suffix("_fcst")
                f1, f2 = _daily_weather(hf, "_l1"), _daily_weather(hf, "_l2")
                fy = (f1, f2)
        # --- збір добових рядків
        days = st.index
        for d in days:
            if d + pd.Timedelta(days=2) not in st.index:
                continue
            r = dict(site=name, time=d, year=y, doy=d.dayofyear)
            for c in wcols:
                r[c] = st.at[d, c]
            r["theta_rz"] = st.at[d, "theta_rz"]
            r["y1"] = st.at[d + pd.Timedelta(days=1), "w_rz"] - st.at[d, "w_rz"]
            r["y2"] = st.at[d + pd.Timedelta(days=2), "w_rz"] - st.at[d, "w_rz"]
            prev1 = d - pd.Timedelta(days=1)
            prev3 = d - pd.Timedelta(days=3)
            r["dw_past1"] = r["w_rz"] - st.at[prev1, "w_rz"] if prev1 in st.index else np.nan
            r["dw_past3"] = r["w_rz"] - st.at[prev3, "w_rz"] if prev3 in st.index else np.nan
            # минула погода (доби d-1, d-2, …) — інтервали, що закінчились до d 00:00
            for k, nd in (("1", 1), ("3", 3), ("7", 7)):
                win = dw.loc[(dw.index >= d - pd.Timedelta(days=nd)) & (dw.index < d)]
                r["P_past" + k] = win["P"].sum() if len(win) == nd else np.nan
                r["ET0_past" + k] = win["ET0"].sum() if len(win) == nd else np.nan
            winT = dw.loc[(dw.index >= d - pd.Timedelta(days=3)) & (dw.index < d)]
            r["T_past3"] = winT["Tmean"].mean() if len(winT) else np.nan
            # погода на горизонт: факт (ideal)
            for j in (1, 2):
                dd = d + pd.Timedelta(days=j - 1)
                for v in ("P", "ET0", "Tmax", "RH", "SW", "U"):
                    r[f"{v}_{j}_ideal"] = dw.at[dd, v] if dd in dw.index else np.nan
                    if fy is not None:
                        src = fy[0] if j == 1 else fy[1]
                        r[f"{v}_{j}_fcst"] = src.at[dd, v] if dd in src.index else np.nan
            if d in ph.index:
                for c in ph.columns:
                    r[c] = ph.at[d, c]
            if phf is not None and d in phf.index:
                for c in phf.columns:
                    r[c] = phf.at[d, c]
            rows.append(r)
        if verbose:
            print(f"  {name} {y}: {sum(1 for r in rows if r['year'] == y)} діб", flush=True)
    df = pd.DataFrame(rows)
    df["theta_range"] = b["theta_rz"][1] - b["theta_rz"][0]
    df["elevation"] = meta.get("elevation")
    df["lat"] = meta.get("lat")
    return df, b


def main():
    io.PROCESSED.mkdir(parents=True, exist_ok=True)
    out, bounds = [], {}
    t0 = time.time()
    for name in io.UA_SITES + [io.FR_SITE]:
        if not (io.RAW / f"{name}_wx.csv").exists():
            print(f"  ! {name}: немає даних, пропускаю")
            continue
        df, b = build_site(name)
        out.append(df)
        bounds[name] = b
        print(f"{name}: готово ({time.time() - t0:.0f} с)", flush=True)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(io.PROCESSED / "daily.csv.gz", index=False)
    (io.PROCESSED / "bounds.json").write_text(json.dumps(bounds, indent=1), encoding="utf-8")
    print(f"Збережено: {len(df)} рядків → {io.PROCESSED / 'daily.csv.gz'}")


if __name__ == "__main__":
    main()
