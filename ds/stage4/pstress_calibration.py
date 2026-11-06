"""
Етап 4, частина A (1.11): чи можна вірити «ймовірності стресу за 48 год», яку видає HydroBrain?

Для кожної доби сезону (15 травня – 30 вересня) 2024–2025 на 14 точках України:
  * стан поля (дефіцит) задається з ERA5-Land, як від ідеального датчика: Dr = (1 − w_0–100) · TAW(корені);
  * HydroBrain (кукурудза, чорнозем, крапельний полив) ухвалює рішення за СПРАВЖНІМ архівним прогнозом погоди
    (Монте-Карло 100 сценаріїв) — двічі: лише фізика і фізика + ML-ядро (як у застосунку);
  * «подія» — за фактом ERA5-Land дефіцит перетнув поріг стресу RAW протягом 48 год.
Беруться лише доби, коли стресу ще немає (саме тоді ймовірність визначає «поливати / ні»).
Оцінка: діаграма надійності, Brier score, ECE; ізотонічне перекалібрування з перехресною перевіркою за роками
(навчання на 2024 → перевірка на 2025 і навпаки).

Спрощення: ML-ядро використовує модель шару 0–100 см незалежно від глибини коренів (стан теж з 0–100 см).
Запуск: python -m ds.stage4.pstress_calibration   (≈ 20–40 хв)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from ..stage2 import data_io as io  # noqa: E402
from . import common as c  # noqa: E402
from hydrobrain import ml_forecast as mlf  # noqa: E402
from hydrobrain.balance import WaterBalanceEngine  # noqa: E402
from hydrobrain.decision import decide  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
N_MC = 100
SIGMA_SENSOR_REL = 0.03        # похибка «ідеального датчика» у відносних одиницях → σ дефіциту


def run_site(name, core, rng_seed=0):
    meta = io.site_meta(name)
    h_all = io.load_hourly(name)
    b = io.scale_bounds(h_all, io.TRAIN_YEARS)
    rows = []
    for y in c.YEARS:
        P = c.resolve_params(meta["lat"], meta["lon"], meta["elevation"], sowing=f"{y}-04-25")
        P["mc_samples"] = N_MC
        P["pstress_calibration"] = False  # fit against raw MC probabilities, including on reruns
        eng = WaterBalanceEngine(P)
        hy = c.load_site_year(name, y)
        if not any(k.endswith("_l1") for k in hy.columns):
            continue
        hy["w"] = io.to_rel(hy["theta_rz"], b["theta_rz"])
        dvs = c.dvs_series(hy, P)
        S_full = eng.prepare(list(hy["time_utc"]), c.weather_dict(hy), dvs, None, None, tz_offset_h=0.0)
        zr = np.asarray(S_full.zr)
        taw = np.array([eng.taw(z) for z in zr])
        mid = np.nonzero((hy["time"].dt.hour == 0).to_numpy())[0]
        for i0 in mid:
            t0 = hy["time"].iloc[i0]
            if not (pd.Timestamp(f"{y}-05-15") <= t0 <= pd.Timestamp(f"{y}-09-30")) or i0 + 49 > len(hy):
                continue
            times, fw, actual, eto_f = c.forecast_window(hy, i0, 48)
            S = eng.prepare(times, fw, dvs[i0 + 1:i0 + 49], None, None, tz_offset_h=0.0)
            w0 = float(hy["w"].iloc[i0])
            dr_now = float(np.clip((1 - w0) * taw[i0], -eng.sat_unit * zr[i0], taw[i0]))
            state = dict(dr=dr_now, de=eng.rew, x=0.0, fw=1.0, zr=float(zr[i0]),
                         du=max(dr_now, 0.0) * min(0.3, zr[i0]) / zr[i0])
            sig = SIGMA_SENSOR_REL * taw[i0]
            prob = np.full(48, np.nan)
            d_phys = decide(eng, S, state, sig, fw["precipitation"], prob, P, rng=np.random.default_rng(rng_seed))
            raw = d_phys["state"]["raw"]
            if dr_now >= raw:
                continue
            # ML-ядро: природна зміна на 24/48 год → зсув траєкторій (як у застосунку)
            bundle = c.ml_features_bundle(hy, i0, eto_f, fw)
            w24 = float(hy["w"].iloc[i0 - 24]) if i0 >= 24 else None
            w72 = float(hy["w"].iloc[i0 - 72]) if i0 >= 72 else None
            dw = c.ml_dw(core, "RZ", c.utc(t0), w0, w24, w72, bundle)
            st1 = eng.init_state(1, dr_now, zr0=float(zr[i0]), du0=np.array([state["du"]]))
            _, rec = eng.run(S, st1, np.nan_to_num(fw["precipitation"]), np.zeros(48), i0=0, i1=48)
            dphys = {1: float(rec["dr"][0, 23]), 2: float(rec["dr"][0, 47])}
            cc = {k: (dr_now - dw[k] * taw[i0]) - dphys[k] for k in (1, 2)}
            sd = {k: core.sigma("RZ", k) * taw[i0] for k in (1, 2)}
            d_ml = decide(eng, S, state, sig, fw["precipitation"], prob, P, rng=np.random.default_rng(rng_seed),
                          dr_shift=mlf.shift_profile(48, cc[1], cc[2]), dr_shift_sd=mlf.sd_profile(48, sd[1], sd[2]))
            # правда: дефіцит за ERA5-Land у наступні 48 год
            w_next = hy["w"].iloc[i0 + 1:i0 + 49].to_numpy()
            dr_next = (1 - w_next) * taw[i0 + 1:i0 + 49]
            event = bool(np.any(dr_next > raw))
            rows.append(dict(site=name, year=y, time=t0, p_phys=d_phys["risk"]["p_stress_48h"],
                             p_ml=d_ml["risk"]["p_stress_48h"], event=event, dr_now=dr_now, raw=raw,
                             rain_fcst=float(np.nansum(fw["precipitation"])),
                             rain_act=float(np.nansum(actual["precipitation"]))))
    return rows


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for k in range(bins):
        m = idx == k
        if m.sum():
            out.append(dict(bin=k, p_mean=float(p[m].mean()), freq=float(y[m].mean()), n=int(m.sum())))
    t = pd.DataFrame(out)
    ece = float(np.sum(t["n"] * np.abs(t["p_mean"] - t["freq"])) / len(p)) if len(t) else np.nan
    return t, ece


def scores(p, y):
    t, ece = reliability(p, y)
    return dict(brier=float(np.mean((p - y) ** 2)), ece=ece, base_rate=float(y.mean()), n=int(len(y)),
                # рішення: «не поливати», якщо p < 0.5 — як часто стрес усе ж стався
                stress_when_skip=float(y[p < 0.5].mean()) if (p < 0.5).any() else np.nan,
                stress_when_act=float(y[p >= 0.5].mean()) if (p >= 0.5).any() else np.nan,
                share_act=float((p >= 0.5).mean()))


def analyse(df):
    y = df["event"].to_numpy(float)
    res, curves, iso_all = {}, {}, {}
    for col in ("p_phys", "p_ml"):
        p = df[col].to_numpy(float)
        res[col] = scores(p, y)
        curves[col] = reliability(p, y)[0]
        # перехресна перевірка за роками
        pc = np.full(len(p), np.nan)
        for yr in c.YEARS:
            tr, te = (df["year"] != yr).to_numpy(), (df["year"] == yr).to_numpy()
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p[tr], y[tr])
            pc[te] = iso.predict(p[te])
        res[col + "_recal_cv"] = scores(pc, y)
        curves[col + "_recal_cv"] = reliability(pc, y)[0]
        iso_all[col] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
    return res, curves, iso_all


def plot(curves, res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, title in ((axes[0], "p_phys", "Лише фізика"), (axes[1], "p_ml", "Фізика + ML-ядро (застосунок)")):
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--", label="ідеальна надійність")
        for key, colr, lab in ((col, "#C77C48", "як є"), (col + "_recal_cv", "#3A8F6B", "після перекалібрування (CV)")):
            t = curves[key]
            ax.plot(t["p_mean"], t["freq"], "o-", color=colr,
                    label=f"{lab}: ECE {res[key]['ece']:.3f}, Brier {res[key]['brier']:.3f}")
            for _, r in t.iterrows():
                ax.annotate(str(int(r.n)), (r.p_mean, r.freq), fontsize=6, color=colr, xytext=(3, -8),
                            textcoords="offset points")
        ax.set_xlabel("ймовірність стресу за 48 год (прогноз HydroBrain)")
        ax.set_ylabel("частка діб, коли стрес справді настав")
        ax.set_title(title)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Надійність ймовірності стресу: 14 точок України, 2024–2025, справжній прогноз погоди")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    core, _ = mlf.MLCore.load_for_evaluation()
    t0 = time.time()
    rows = []
    for s in io.UA_SITES:
        rows += run_site(s, core)
        print(f"  {s}: {len(rows)} рішень ({time.time() - t0:.0f} с)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "pstress_decisions.csv.gz", index=False)
    res, curves, iso = analyse(df)
    plot(curves, res, RES / "fig1_reliability.png")
    calib = {k: dict(x=[float(v) for v in m.X_thresholds_], y=[float(v) for v in m.y_thresholds_]) for k, m in iso.items()}
    (RES / "pstress_metrics.json").write_text(json.dumps(dict(metrics=res, isotonic=calib), indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
