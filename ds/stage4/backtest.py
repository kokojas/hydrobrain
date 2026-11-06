"""
Етап 4, частина B (1.11): бектест поливу — чи краще HydroBrain поливає, ніж прості стратегії.

«Справжнє поле» (віртуальне): фізична модель, у якої для кожної точки підібрано запас води (×TAW) і транспірацію
(×Kc) так, щоб БЕЗ поливу вона якомога ближче відтворювала ERA5-Land 2016–2021 (див. fit_truth). HydroBrain цих
параметрів не знає: працює з довідниковими (кукурудза, чорнозем типовий, крапельний полив), отримує щоденний
показ «датчика» з поля з шумом і ухвалює рішення за СПРАВЖНІМ архівним прогнозом погоди 2024–2025.
Поле живе за фактичною погодою.

Стратегії (полив 15 травня – 15 вересня, оцінка 15 травня – 30 вересня):
  rainfed     — без поливу (довідково);
  schedule    — 25 мм кожні 7 діб (типова практика);
  threshold   — поріг за датчиком: якщо відносна вологість < 0.5, полити до польової вологоємності;
  hb_physics  — HydroBrain, лише фізика;
  hb_ml       — HydroBrain, фізика + ML-ядро (як у застосунку), поріг рішення p_skip = 0.5;
  hb_ml_p03 / hb_ml_p07 — те саме з порогом 0.3 / 0.7 (крива «вода ↔ стрес»);
  oracle      — HydroBrain, що знає справжні параметри поля, справжній стан і майбутню погоду (верхня межа).
Метрики: вода (мм, кількість поливів), години/доби стресу (Ks < 1), середній Ks, втрати (дренаж + стік),
орієнтовна втрата врожаю 1 − Ya/Ym = Ky·(1 − ΣT/ΣTпот), Ky = 1.25 для кукурудзи (FAO-33).

Чесне застереження: це «світ фізики» — чесніший за прогін моделі самої на собі (HydroBrain не знає справжніх
параметрів і стану, рішення — за прогнозом), але не реальне поле.
Запуск: python -m ds.stage4.backtest   (≈ 40–60 хв)
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

from ..stage2 import data_io as io  # noqa: E402
from ..stage2 import physics_runs as pr  # noqa: E402
from . import common as c  # noqa: E402
from hydrobrain import ml_forecast as mlf  # noqa: E402
from hydrobrain.balance import WaterBalanceEngine  # noqa: E402
from hydrobrain.decision import decide  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
N_MC = 60
SENSOR_NOISE = 0.02          # шум датчика, відносні одиниці
KY_MAIZE = 1.25
STRATEGIES = ["rainfed", "schedule", "threshold", "hb_physics", "hb_ml", "hb_ml_p03", "hb_ml_p07", "oracle"]
EXTRA = ["hb_physics_cal", "hb_ml_cal"]      # 1.11: з перекаліброваною ймовірністю стресу (як у застосунку з 1.11)
ALL = STRATEGIES[:-1] + EXTRA + ["oracle"]
UK = dict(rainfed="Без поливу", schedule="За графіком (25 мм / 7 діб)", threshold="Поріг за датчиком",
          hb_physics="HydroBrain: лише фізика", hb_ml="HydroBrain: фізика + ML", hb_ml_p03="HydroBrain ML, поріг 0.3",
          hb_ml_p07="HydroBrain ML, поріг 0.7", oracle="Оракул (знає все)",
          hb_physics_cal="HydroBrain: фізика, калібр. ймовірність", hb_ml_cal="HydroBrain 1.11: ML + калібр. ймовірність")
TAW_GRID = [0.6, 0.8, 1.0, 1.25, 1.5, 1.8]
K_GRID = [0.6, 0.75, 0.9, 1.05, 1.2]


# ---------------------------------------------------------------------------- «справжнє поле»
def fit_truth(name):
    """Підбір ×TAW і ×Kc, за яких фізика без поливу найкраще відтворює ERA5-Land (2016–2021, щотижневі старти)."""
    meta = io.site_meta(name)
    h = io.load_hourly(name)
    b = io.scale_bounds(h, io.TRAIN_YEARS)
    eng = pr.make_engine(meta["lat"], meta["lon"], meta["elevation"])
    grid = [(a, k) for a in TAW_GRID for k in K_GRID]
    tm = np.array([g[0] for g in grid])
    km = np.array([g[1] for g in grid])
    N = len(grid)
    err = np.zeros(N)
    cnt = 0
    for y in io.TRAIN_YEARS:
        hy = h[h["year"] == y].reset_index(drop=True)
        times = [c.utc(t) for t in hy["time"]]
        w = io.to_rel(hy["theta_rz"], b["theta_rz"])
        S = eng.prepare(times, pr._weather_dict(hy), np.full(len(times), 0.5), None, None, tz_offset_h=0.0)
        taw0 = eng.taw(pr.ZR)
        is0 = (hy["time"].dt.hour == 0).to_numpy()
        day = np.cumsum(is0)
        st = eng.init_state(N, np.clip((1 - w[0]) * taw0 * tm, 0, taw0 * tm), zr0=pr.ZR)
        acc = dict(e=np.zeros(N), n=0)

        def cb(i, s, flux):
            if not is0[i]:
                return
            wm = 1.0 - s["dr"] / (taw0 * tm)
            if day[i] % 7 != 0:
                acc["e"] += np.abs(wm - w[i])
                acc["n"] += 1
            else:
                s["dr"][:] = np.clip((1 - w[i]) * taw0 * tm, -eng.sat_unit * pr.ZR, taw0 * tm)
                if "du" in s:
                    s["du"][:] = np.clip(s["dr"], 0, None) * 0.3
        eng.run(S, st, np.nan_to_num(hy["precipitation"].to_numpy(float)), np.zeros(len(times)),
                taw_mult=tm, learn_k=km, callback=cb, record=False)
        err += acc["e"]
        cnt += acc["n"]
    mae = err / max(cnt, 1)
    j = int(np.argmin(mae))
    k1 = grid.index((1.0, 1.05)) if (1.0, 1.05) in grid else j
    return dict(taw_mult=grid[j][0], k_mult=grid[j][1], mae=float(mae[j]), mae_default=float(mae[grid.index((1.0, 0.9))]))


# ---------------------------------------------------------------------------- один сезон
def season(name, year, truth, core, strategies=None):
    meta = io.site_meta(name)
    P = c.resolve_params(meta["lat"], meta["lon"], meta["elevation"], sowing=f"{year}-04-25")
    P["mc_samples"] = N_MC
    engD = WaterBalanceEngine(P)                                   # HydroBrain: довідникові параметри
    Q, kT = c.truth_params(P, truth["taw_mult"], truth["k_mult"])
    engT = WaterBalanceEngine(Q)                                   # «справжнє поле»
    hy = c.load_site_year(name, year)
    h_all = io.load_hourly(name)
    b = io.scale_bounds(h_all, io.TRAIN_YEARS)
    w_e5 = io.to_rel(hy["theta_rz"], b["theta_rz"])
    times = list(hy["time_utc"])
    dvs = c.dvs_series(hy, P)
    ST = engT.prepare(times, c.weather_dict(hy), dvs, None, None, tz_offset_h=0.0)
    SD_full = engD.prepare(times, c.weather_dict(hy), dvs, None, None, tz_offset_h=0.0)
    rain = np.nan_to_num(hy["precipitation"].to_numpy(float))
    zrT = np.asarray(ST.zr)
    tawT = np.array([engT.taw(z) for z in zrT])
    tawD = np.array([engD.taw(z) for z in np.asarray(SD_full.zr)])
    tt = hy["time"]
    i_start = int(np.argmax((tt >= pd.Timestamp(f"{year}-05-15")).to_numpy()))
    i_end = int(np.argmax((tt >= pd.Timestamp(f"{year}-10-01")).to_numpy())) or len(hy) - 1
    irr_last = pd.Timestamp(f"{year}-09-15")
    # стан «справжнього поля» до 15 травня — без поливу, старт з ERA5-Land
    st0 = engT.init_state(1, float(np.clip((1 - w_e5[0]) * tawT[0], 0, tawT[0])), zr0=float(zrT[0]))
    st0, _ = engT.run(ST, st0, rain, np.zeros(len(hy)), learn_k=kT, i0=0, i1=i_start + 1, record=False)
    rng = np.random.default_rng(hash((name, year)) % 2 ** 32)
    noise = rng.normal(0, SENSOR_NOISE, len(hy))
    out = {}
    for strat in (strategies or STRATEGIES):
        st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st0.items()}
        m = dict(irr=0.0, n_irr=0, stress_h=0, ks_sum=0.0, hours=0, t=0.0, tpot=0.0, loss=0.0)
        hist_w, irr_times = {}, []
        for i0 in range(i_start, i_end - 24):
            if tt.iloc[i0].hour != 0:
                continue
            day = tt.iloc[i0]
            irr = np.zeros(len(hy))
            w_true = 1.0 - float(st["dr"][0]) / tawT[i0]
            w_sens = w_true + noise[i0]
            hist_w[i0] = w_sens
            gross, h_off = 0.0, 0
            can = day <= irr_last and strat != "rainfed"
            if can and strat == "schedule":
                if (day - pd.Timestamp(f"{year}-05-15")).days % 7 == 0:
                    gross, h_off = 25.0, 6
            elif can and strat == "threshold":
                if w_sens < 0.5:
                    net = min((1 - w_sens) * tawD[i0], engD.method.get("max_mm", 40.0))
                    gross, h_off = net / engD.method["ea"], 6
            elif can:
                gross, h_off = hb_decide(strat, engD, engT, kT, hy, i0, dvs, w_sens, w_true, tawD, tawT, P, Q,
                                         core, hist_w, irr_times, st, zrT)
            if gross > 0 and h_off < 24:
                irr[i0 + 1 + h_off] = gross
                m["irr"] += gross
                m["n_irr"] += 1
                irr_times.append(i0 + 1 + h_off)
            st, rec = engT.run(ST, st, rain, irr, learn_k=kT, i0=i0 + 1, i1=i0 + 25)
            ks = rec["ks"][0]
            m["stress_h"] += int(np.sum(ks < 0.999))
            m["ks_sum"] += float(ks.sum())
            m["hours"] += len(ks)
            t = rec["t"][0]
            m["t"] += float(t.sum())
            m["tpot"] += float(np.sum(np.where(ks > 0.05, t / np.maximum(ks, 0.05), 0.0)))
            m["loss"] += float(rec["dp"][0].sum() + rec["ro"][0].sum())
        out[strat] = dict(site=name, year=year, strategy=strat, irrigation_mm=m["irr"], n_irrigations=m["n_irr"],
                          stress_days=m["stress_h"] / 24.0, mean_ks=m["ks_sum"] / max(m["hours"], 1),
                          losses_mm=m["loss"],
                          yield_loss_pct=100 * KY_MAIZE * (1 - m["t"] / max(m["tpot"], 1e-6)))
    return list(out.values())


def hb_decide(strat, engD, engT, kT, hy, i0, dvs, w_sens, w_true, tawD, tawT, P, Q, core, hist_w, irr_times, st, zrT):
    oracle = strat == "oracle"
    eng = engT if oracle else engD
    times, fw, actual, eto_f = c.forecast_window(hy, i0, 48)
    wx = actual if oracle else fw
    S = eng.prepare(times, wx, dvs[i0 + 1:i0 + 49], None, None, tz_offset_h=0.0)
    if oracle:
        dr = float(st["dr"][0])
        sig = 1.0
    else:
        dr = float(np.clip((1 - w_sens) * tawD[i0], -eng.sat_unit * zrT[i0], tawD[i0]))
        sig = 0.03 * tawD[i0]
    state = dict(dr=dr, de=float(st["de"][0]), x=0.0, fw=float(st["fw"][0]), zr=float(zrT[i0]),
                 du=max(dr, 0.0) * min(0.3, zrT[i0]) / zrT[i0])
    PP = dict(Q if oracle else P)
    PP["pstress_calibration"] = strat.endswith("_cal")       # базові стратегії — без перекалібрування (як до 1.11)
    if strat == "hb_ml_p03":
        PP["decision_thresholds"] = {"p_skip": 0.3}
    if strat == "hb_ml_p07":
        PP["decision_thresholds"] = {"p_skip": 0.7}
    prob = np.where(np.asarray(wx["precipitation"]) > 0.1, 100.0, 0.0) if oracle else np.full(48, np.nan)
    shift = sd = None
    if strat.startswith("hb_ml"):          # hb_ml, hb_ml_p03/p07, hb_ml_cal — з ML-ядром
        recent = lambda a, b: any(i0 - a < j <= i0 - b for j in irr_times)
        w24 = None if recent(24, 0) else hist_w.get(i0 - 24)
        w72 = None if recent(72, 0) else hist_w.get(i0 - 72)
        bundle = c.ml_features_bundle(hy, i0, eto_f, fw)
        dw = c.ml_dw(core, "RZ", c.utc(hy["time"].iloc[i0]), w_sens, w24, w72, bundle)
        st1 = eng.init_state(1, dr, zr0=float(zrT[i0]), du0=np.array([state["du"]]))
        _, rec = eng.run(S, st1, np.nan_to_num(fw["precipitation"]), np.zeros(48), i0=0, i1=48)
        cc = {k: (dr - dw[k] * tawD[i0]) - float(rec["dr"][0, 24 * k - 1]) for k in (1, 2)}
        shift = mlf.shift_profile(48, cc[1], cc[2])
        sd = mlf.sd_profile(48, core.sigma("RZ", 1) * tawD[i0], core.sigma("RZ", 2) * tawD[i0])
    d = decide(eng, S, state, sig, np.asarray(wx["precipitation"]), prob, PP, k_learn=(kT if oracle else 1.0),
               rng=np.random.default_rng(i0), dr_shift=shift, dr_shift_sd=sd)
    if d["action"] == "irrigate_now" and d["amount"]:
        return float(d["amount"]["gross_mm"]), 0
    if d["action"] == "postpone" and d["amount"] and d["t_idx"] is not None and d["t_idx"] < 24:
        return float(d["amount"]["gross_mm"]), int(d["t_idx"])
    return 0.0, 0


# ---------------------------------------------------------------------------- звіт
def report(df, truths):
    order = [s for s in ALL if s in set(df["strategy"])]
    agg = df.groupby("strategy")[["irrigation_mm", "n_irrigations", "stress_days", "mean_ks", "losses_mm",
                                  "yield_loss_pct"]].mean().loc[order]
    agg.to_csv(RES / "backtest_summary.csv")
    main_s = [x for x in ["schedule", "threshold", "hb_physics", "hb_ml", "hb_ml_cal", "oracle"] if x in agg.index]
    col = dict(schedule="#C77C48", threshold="#B39DDB", hb_physics="#E0A878", hb_ml="#8FA8D0", hb_ml_cal="#2E5C9A",
               oracle="#3A8F6B")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    y = np.arange(len(main_s))[::-1]
    for ax, key, lab, fmt in ((axes[0], "irrigation_mm", "полив за сезон, мм", "{:.0f}"),
                              (axes[1], "stress_days", "доби стресу (Ks < 1) за сезон", "{:.1f}")):
        v = agg.loc[main_s, key].to_numpy()
        ax.barh(y, v, color=[col[m] for m in main_s])
        for yy, vv in zip(y, v):
            ax.text(vv, yy, " " + fmt.format(vv), va="center", fontsize=9)
        ax.set_yticks(y, [UK[m] for m in main_s] if ax is axes[0] else [""] * len(main_s))
        ax.set_xlabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Бектест поливу: 14 точок × 2024–2025, кукурудза, рішення за справжнім прогнозом погоди "
                 f"(без поливу: {agg.loc['rainfed', 'stress_days']:.0f} діб стресу)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "fig2_water_vs_stress.png", dpi=150)
    plt.close(fig)
    cur = [x for x in ["hb_ml_p03", "hb_ml", "hb_ml_p07"] if x in agg.index]
    if len(cur) == 3:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        pts = agg.loc[cur]
        ax.plot(pts.irrigation_mm, pts.stress_days, "o-", color="#5B7DB1", label="HydroBrain ML: поріг 0.3 / 0.5 / 0.7")
        for m, t in zip(cur, ["0.3", "0.5", "0.7"]):
            ax.annotate(f"поріг {t}", (pts.loc[m].irrigation_mm, pts.loc[m].stress_days), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
        for m, mk in (("hb_ml_cal", "s"), ("oracle", "*"), ("threshold", "^"), ("schedule", "D")):
            if m in agg.index:
                ax.plot(agg.loc[m].irrigation_mm, agg.loc[m].stress_days, mk, ms=9, color=col[m], label=UK[m])
        ax.set_xlabel("полив за сезон, мм")
        ax.set_ylabel("доби стресу за сезон")
        ax.set_title("Компроміс «вода ↔ стрес» і вибір порогу рішення")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(RES / "fig3_threshold_tradeoff.png", dpi=150)
        plt.close(fig)
    lines = ["# Бектест поливу (віртуальне поле, 14 точок × 2024–2025, кукурудза, крапельний полив)\n\n",
             "Середнє за 28 сезонів. Втрата врожаю — орієнтовна, за FAO-33 (Ky = 1.25).\n\n",
             "| Стратегія | Полив, мм | Поливів | Діб стресу | Середній Ks | Втрати (дренаж+стік), мм | Втрата врожаю, % |\n",
             "|---|---|---|---|---|---|---|\n"]
    for s in order:
        r = agg.loc[s]
        lines.append(f"| {UK[s]} | {r.irrigation_mm:.0f} | {r.n_irrigations:.1f} | {r.stress_days:.1f} | "
                     f"{r.mean_ks:.3f} | {r.losses_mm:.0f} | {r.yield_loss_pct:.1f} |\n")
    lines.append("\n## Параметри «справжнього поля» (підібрані під ERA5-Land 2016–2021)\n\n| Точка | ×TAW | ×Kc | MAE (відн.) | MAE довідник |\n|---|---|---|---|---|\n")
    for k, v in truths.items():
        lines.append(f"| {k} | {v['taw_mult']} | {v['k_mult']} | {v['mae']:.3f} | {v['mae_default']:.3f} |\n")
    (RES / "backtest_summary.md").write_text("".join(lines), encoding="utf-8")
    return agg


def extra():
    """Догнати стратегії з перекаліброваною ймовірністю (параметри «справжнього поля» — з попереднього запуску)."""
    core, _ = mlf.MLCore.load_for_evaluation()
    truths = json.loads((RES / "backtest_truth_params.json").read_text(encoding="utf-8"))
    df = pd.read_csv(RES / "backtest_seasons.csv")
    df = df[~df["strategy"].isin(EXTRA)]
    rows = []
    t0 = time.time()
    for s, tr in truths.items():
        for y in c.YEARS:
            rows += season(s, y, tr, core, EXTRA)
        print(f"  {s}: додаткові стратегії ({time.time() - t0:.0f} с)", flush=True)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(RES / "backtest_seasons.csv", index=False)
    print(report(df, truths).round(2).to_string())


def main(sites=None):
    RES.mkdir(parents=True, exist_ok=True)
    core, _ = mlf.MLCore.load_for_evaluation()
    t0 = time.time()
    truths, rows = {}, []
    for s in sites or io.UA_SITES:
        truths[s] = fit_truth(s)
        for y in c.YEARS:
            rows += season(s, y, truths[s], core)
        print(f"  {s}: {truths[s]} ({time.time() - t0:.0f} с)", flush=True)
        pd.DataFrame(rows).to_csv(RES / "backtest_seasons.csv", index=False)
    (RES / "backtest_truth_params.json").write_text(json.dumps(truths, indent=1), encoding="utf-8")
    df = pd.DataFrame(rows)
    print(report(df, truths).round(2).to_string())


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--extra"]:
        extra()
    else:
        main(sys.argv[1:] or None)
