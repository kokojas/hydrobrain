"""
Крива навчання «нове поле»: скільки даних датчика потрібно, щоб модель стала корисною.

Сценарій: на полі щойно поставили датчик. Через X діб (X = 7, 14, 30, 60, 120, 240) калібруємо моделі
ЛИШЕ на цих X добах і далі прогнозуємо на тестових роках 2018–2022. Усе, що модель знає про місце, —
з вікна: масштаб датчика (1-й/99-й перцентилі вікна), нахил калібрування фізики, навчальні приклади ML.
Старт вікна: 1 квітня і 1 червня 2016 та 2017 років (4 старти × 3 станції = 12 «нових полів» на точку
кривої), щоб результат не залежав від одного вдалого чи невдалого вікна.

Моделі на кривій (ціль — профіль 0–30 см, зміна за 1 і 2 доби, MAE у % об.):
  persistence      — нічого не потребує;
  physics_cal_win  — фізика (ґрунт HWSD) + масштаб і лінійне калібрування з вікна  [M1c «з нуля»];
  ml_ua_win        — ML, навчена на ERA5-Land України; з вікна лише масштаб датчика;
  ml_win           — ML, навчена лише на добах вікна.
Горизонтальні орієнтири (з Етапу 3, 5 сезонів історії): M5 ML на датчиках, M1c фізика + калібрування.

Запуск: python -m ds.stage3.learning_curve   (≈ 10 хв)
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..stage2 import data_io as io  # noqa: E402
from ..stage2 import physics_runs as pr  # noqa: E402
from ..stage2.build_dataset import _daily_weather  # noqa: E402
from . import build as b3  # noqa: E402
from .experiment import RES, feats, gbm, prep  # noqa: E402

WINDOWS = [7, 14, 30, 60, 120, 240]
STARTS = ["2016-04-01", "2016-06-01", "2017-04-01", "2017-06-01"]
MIN_RANGE = 0.03          # якщо у вікні вологість майже не змінювалась, масштаб розширюється до 3 % об.
MODELS = ["persistence", "physics_cal_win", "ml_ua_win", "ml_win"]
UK = dict(persistence="Персистентність", physics_cal_win="Фізика + калібрування з вікна",
          ml_ua_win="ML з України (ERA5-Land), масштаб з вікна", ml_win="ML лише на даних вікна")
COL = dict(persistence="#9E9E9E", physics_cal_win="#E0A878", ml_ua_win="#5B7DB1", ml_win="#2E5C9A")


TCFG = dict(T2=dict(col="s_prof", zr=b3.PROFILE_BOTTOM, uk="Профіль 0–30 см"),
            T3=dict(col="s_deep", zr=b3.DEEP_BOTTOM, uk="Профіль 0–55 см"))


def station_frame(st, h, dw, tg="T2"):
    sh, _ = b3.sensor_hourly(st)
    mid = b3.at_midnight(sh)
    df = pd.DataFrame(index=mid.index)
    df["th"] = mid[TCFG[tg]["col"]]
    df = df[df.index.isin(dw.index)]
    df = b3.add_targets(df, "th", tg)            # y_<tg>_h, dw_<tg>_past* — у абсолютних θ
    df = b3.add_weather_feats(df, dw)
    df["doy"] = df.index.dayofyear
    df["year"] = df.index.year
    return df, mid


def physics_rel(eng, h, mid, lo, hi, years, col="s_prof"):
    """Прогін фізики з відносним масштабом (lo, hi): повертає Δθ-прогноз на 1 і 2 доби."""
    rng = hi - lo
    out = []
    for y in years:
        hy = h[h["year"] == y].reset_index(drop=True)
        if len(hy) < 24 * 20:
            continue
        tt = hy["time_naive"]
        ww = mid.reindex(tt.dt.floor("D"))
        ww.index = tt.index
        is0 = (tt.dt.hour == 0).to_numpy()
        wp_ = np.where(is0, (ww[col] - lo) / rng, np.nan)
        ws_ = np.where(is0, (ww["s_surf"] - lo) / rng, np.nan)
        for sc in (0, 1):
            d, _ = pr.run_season(eng, hy, wp_, ws_, sc)
            if len(d):
                out.append(d)
    d = pd.concat(out).drop_duplicates("time").set_index("time")
    d.index = d.index.tz_localize(None)
    w_in = (mid[col].reindex(d.index) - lo) / rng
    return pd.DataFrame({f"p{j}": (d[f"wphys_{j}"] - w_in) * rng for j in (1, 2)})


def run(tg="T2"):
    t0 = time.time()
    col = TCFG[tg]["col"]
    meta = io.site_meta(io.FR_SITE)
    h = io.load_hourly(io.FR_SITE)
    h["time_naive"] = h["time"].dt.tz_localize(None) if h["time"].dt.tz is not None else h["time"]
    dw = _daily_weather(h.assign(time=h["time_naive"]))
    ua = pd.read_csv(io.PROCESSED / "stage3_ua.csv.gz", parse_dates=["time"])
    ml_ua = {}
    for hh in (1, 2):
        u = prep(ua, tg, hh).dropna(subset=["y"])
        ml_ua[hh] = gbm(feats(hh), True).fit(u[feats(hh)], u["y"])
    rows = []
    for st in b3.STATIONS:
        df, mid = station_frame(st, h, dw, tg)
        eng = pr.make_engine(meta["lat"], meta["lon"], meta["elevation"], soil=b3.hwsd_soil(st), zr=TCFG[tg]["zr"])
        test = df[df["year"].isin(b3.TEST_Y)]
        for start in STARTS:
            s0 = pd.Timestamp(start)
            if df.loc[df.index >= s0, "th"].first_valid_index() is None:
                continue
            for X in WINDOWS:
                win = df[(df.index >= s0) & (df.index < s0 + pd.Timedelta(days=X))]
                win = win[win["th"].notna()]
                if len(win) < 3:
                    continue
                lo, hi = np.nanpercentile(win["th"], 1), np.nanpercentile(win["th"], 99)
                if hi - lo < MIN_RANGE:
                    c = (hi + lo) / 2
                    lo, hi = c - MIN_RANGE / 2, c + MIN_RANGE / 2
                years = sorted(set(win["year"]) | set(b3.TEST_Y))
                ph = physics_rel(eng, h, mid, lo, hi, years, col)
                for hh in (1, 2):
                    wtr = win.join(ph).dropna(subset=[f"y_{tg}_{hh}", f"p{hh}"])
                    te = test.join(ph).dropna(subset=[f"y_{tg}_{hh}", f"p{hh}", "th"])
                    if len(te) == 0:
                        continue
                    y = te[f"y_{tg}_{hh}"].to_numpy()
                    P = {"persistence": np.zeros(len(te))}
                    if len(wtr) >= 3 and np.std(wtr[f"p{hh}"]) > 0:
                        a, c = np.polyfit(wtr[f"p{hh}"], wtr[f"y_{tg}_{hh}"], 1)
                    else:
                        a, c = 1.0, 0.0
                    P["physics_cal_win"] = a * te[f"p{hh}"].to_numpy() + c
                    # ML з України: вхід у відносних одиницях вікна, вихід назад у θ
                    rel = te.copy()
                    rel[f"w_{tg}"] = (rel["th"] - lo) / (hi - lo)
                    rel[f"dw_{tg}_past1"] = rel[f"dw_{tg}_past1"] / (hi - lo)
                    rel[f"dw_{tg}_past3"] = rel[f"dw_{tg}_past3"] / (hi - lo)
                    rel[f"y_{tg}_{hh}"] = rel[f"y_{tg}_{hh}"] / (hi - lo)
                    rp = prep(rel, tg, hh)
                    P["ml_ua_win"] = ml_ua[hh].predict(rp[feats(hh)]) * (hi - lo)
                    # ML лише на добах вікна (абсолютні θ)
                    wa = prep(win.assign(**{f"w_{tg}": win["th"]}), tg, hh).dropna(subset=["y"])
                    ta = prep(te.assign(**{f"w_{tg}": te["th"]}), tg, hh)
                    if len(wa) >= 3:
                        P["ml_win"] = gbm(feats(hh), False).fit(wa[feats(hh)], wa["y"]).predict(ta[feats(hh)])
                    else:
                        P["ml_win"] = np.full(len(te), np.nan)
                    for m in MODELS:
                        e = np.abs(P[m] - y) * 100
                        rows.append(dict(station=st, start=start, X=X, h=hh, model=m, n=len(te),
                                         MAE=float(np.nanmean(e)), slope=float(a) if m == "physics_cal_win" else np.nan))
            print(f"  {st} {start}: готово ({time.time() - t0:.0f} с)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / LC_FILE[tg], index=False)
    plot(res, tg)
    return res


LC_FILE = dict(T2="learning_curve.csv", T3="learning_curve_T3.csv")


def plot(res, tg="T2"):
    ref = pd.read_csv(RES / "metrics.csv")
    ref = ref[(ref.station == "усі") & (ref.target == tg)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, hh in zip(axes, (1, 2)):
        r = res[res.h == hh]
        for m in MODELS:
            g = r[r.model == m].groupby("X")["MAE"]
            mean, q1, q3 = g.mean(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(mean.index, mean.values, "o-", color=COL[m], label=UK[m])
            ax.fill_between(mean.index, q1.values, q3.values, color=COL[m], alpha=0.15)
        for m, ls, lab in (("ml_local", "--", "ML на датчиках, 5 сезонів"),
                           ("physics_cal", ":", "Фізика + калібрування, 5 сезонів")):
            v = ref[(ref.model == m) & (ref.h == hh)]["MAE"].iloc[0]
            ax.axhline(v, color="#333333", ls=ls, lw=1, label=lab)
        top = 2.2 * r[r.model == "persistence"]["MAE"].mean()
        ax.set_ylim(0, top)
        off = r[(r.model == "physics_cal_win")].groupby("X")["MAE"].mean()
        for X, v in off.items():
            if v > top:
                ax.annotate(f"{v:.1f}↑", (X, top * 0.97), ha="center", va="top", fontsize=7, color=COL["physics_cal_win"])
        ax.set_xscale("log")
        ax.set_xticks(WINDOWS, [str(w) for w in WINDOWS])
        ax.set_xlabel("діб даних датчика на новому полі")
        ax.set_ylabel("MAE, % об. (тест 2018–2022)")
        ax.set_title(f"{TCFG[tg]['uk']}, горизонт {hh} доб.")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Крива навчання «нове поле»: 3 станції × 4 старти; лінія — середнє, смуга — 25–75%")
    fig.tight_layout()
    fig.savefig(RES / ("fig3_learning_curve.png" if tg == "T2" else f"fig3_learning_curve_{tg}.png"), dpi=150)
    plt.close(fig)
    tab = res.groupby(["h", "model", "X"])["MAE"].mean().unstack("X").round(3)
    (RES / ("learning_curve_summary.md" if tg == "T2" else f"learning_curve_summary_{tg}.md")).write_text(
        f"# Крива навчання «нове поле», {TCFG[tg]['uk']} (MAE, % об., середнє за 12 вікнами)\n\n"
        + tab.to_markdown() + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys as _s
    run(_s.argv[1] if len(_s.argv) > 1 else "T2")
