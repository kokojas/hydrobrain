"""
Експеримент «точка на карті без датчика»: наскільки добре можна оцінити ПОТОЧНИЙ стан ґрунту, не маючи
датчика на полі, — і хто з цим справляється краще: фізика чи ERA5-Land + ML-ядро HydroBrain.

Правда — реальні датчики ISMN (fraye, grandcal, hillan2), тест 2018–2022 (вегетаційні сезони).
Оцінюємо РІВЕНЬ вологості, а не зміну, у відносних одиницях (0 — найсухіше, 1 — найвологіше за
багаторічними перцентилями датчика). Для рішення про полив важливо, чи ґрунт нижче порогу: тут поріг
w < 0.5 (як RAW при p ≈ 0.5); чутливість до порогу 0.4 і 0.6 — у таблиці.

Варіанти оцінки стану (усе, що знає система про точку, — координати і погода):
  climatology   — середня вологість ERA5-Land для цього дня року (2013–2017): «як зазвичай у цю пору»
  physics       — фізика HydroBrain без датчика: старт 20 березня з ERA5-Land, далі баланс увесь сезон
                  (так працювала б версія 1.0 для точки на карті)
  era5l_now     — ERA5-Land у точці на той самий день (ідеал: у реальності дані запізнюються на ~5 діб)
  era5l_lag5_ml — ERA5-Land 5 діб тому, прокручений до сьогодні ML-ядром HydroBrain за фактичною погодою
                  (те, що реально можна зробити в застосунку)
Для порівняння — сценарій «ручний вимір колись був»:
  sensor7_ml / sensor3_ml — показ датчика 7 / 3 доби тому, прокручений ML-ядром до сьогодні.

Запуск: python -m ds.stage3.no_sensor   (≈ 3–5 хв; потрібні stage3_france.csv.gz і моделі hydrobrain/models)
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..stage2 import data_io as io  # noqa: E402
from ..stage2 import physics_runs as pr  # noqa: E402
from . import build as b3  # noqa: E402
from .experiment import RES  # noqa: E402

TARGETS = dict(T2=dict(zr=b3.PROFILE_BOTTOM, core="T2", uk="0–30 см"), T3=dict(zr=b3.DEEP_BOTTOM, core="T3", uk="0–55 см"))
VARIANTS = ["climatology", "physics", "era5l_now", "era5l_lag5_ml", "sensor7_ml", "sensor3_ml"]
UK = dict(climatology="Кліматологія ERA5-Land (день року)", physics="Фізика без датчика (весь сезон)",
          era5l_now="ERA5-Land того ж дня (ідеал)", era5l_lag5_ml="ERA5-Land −5 діб + ML-ядро",
          sensor7_ml="Датчик 7 діб тому + ML-ядро", sensor3_ml="Датчик 3 доби тому + ML-ядро")
COL = dict(climatology="#9E9E9E", physics="#C77C48", era5l_now="#B39DDB", era5l_lag5_ml="#5B7DB1",
           sensor7_ml="#3A8F6B", sensor3_ml="#1F6B4A")
THR = 0.5
MODEL = io.ROOT / "hydrobrain" / "models" / "ml_core_v2.joblib"


def core_features(row: pd.DataFrame, w, w1, w3):
    """Ознаки ML-ядра (горизонт 1 доба) з добових рядків фрейму Етапу 3."""
    x = pd.DataFrame(index=row.index)
    x["w"], x["dw1"], x["dw3"] = w, w - w1, w - w3
    for c in ["P_past1", "P_past3", "P_past7", "ET0_past1", "ET0_past3", "ET0_past7", "T_past3"]:
        x[c] = row[c].to_numpy()
    x["doy_sin"] = np.sin(2 * np.pi * row["doy"].to_numpy() / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * row["doy"].to_numpy() / 365.25)
    for v in ["P", "ET0", "Tmax", "RH", "SW", "U"]:
        x[f"{v}_1"] = row[f"{v}_1"].to_numpy()
    x["P_h"], x["ET0_h"] = x["P_1"], x["ET0_1"]
    return x


def rollout(df: pd.DataFrame, src: str, lag: int, model):
    """Для кожної доби d: стан src на d−lag, прокручений ML-ядром до d за фактичною погодою (крок — 1 доба)."""
    s = df[src]
    days = df.index
    past = lambda k: s.shift(k, freq="D").reindex(days).to_numpy().copy()
    hist = [past(lag + 3), past(lag + 2), past(lag + 1), past(lag)]      # w(t−3), w(t−2), w(t−1), w(t)
    for k in range(lag, 0, -1):
        rows = df.reindex(days - pd.Timedelta(days=k))                   # доба τ = d − k: погода τ → τ+1
        w, w1, w3 = hist[3], hist[2], hist[0]
        x = core_features(rows, w, w1, w3)
        ok = np.isfinite(w) & np.isfinite(x["P_1"].to_numpy())
        dw = np.full(len(w), np.nan)
        if ok.any():
            dw[ok] = model.predict(x[ok])
        hist = hist[1:] + [w + dw]
    return pd.Series(hist[3], index=days)


def physics_free(meta, soil, h, e_w: pd.Series, zr):
    """Фізика без датчика: на старті сезону — ERA5-Land, далі безперервний баланс без жодних поправок."""
    eng = pr.make_engine(meta["lat"], meta["lon"], meta["elevation"], soil=soil, zr=zr)
    taw = float(eng.taw(zr))
    out = {}
    for y in sorted(h["year"].unique()):
        hy = h[h["year"] == y].reset_index(drop=True)
        if len(hy) < 24 * 20:
            continue
        from datetime import timezone
        times = [t.to_pydatetime().replace(tzinfo=timezone.utc) for t in hy["time_naive"]]
        wx = pr._weather_dict(hy)
        S = eng.prepare(times, wx, np.full(len(times), 0.5), None, None, tz_offset_h=0.0)
        w0 = e_w.get(pd.Timestamp(hy["time_naive"].iloc[0]).floor("D"), 0.85)
        w0 = 0.85 if not np.isfinite(w0) else w0
        st = eng.init_state(1, float(np.clip((1 - w0) * taw, -eng.sat_unit * zr, taw)), zr0=zr)
        _, rec = eng.run(S, st, np.nan_to_num(wx["precipitation"]), np.zeros(len(times)), learn_k=1.0)
        dr = rec["dr"][0]
        for i, t in enumerate(hy["time_naive"]):
            if t.hour == 0:
                out[t] = 1.0 - dr[i] / taw
    return pd.Series(out)


def score(true, est, rng_theta):
    ok = np.isfinite(true) & np.isfinite(est)
    t, e = true[ok], est[ok]
    r = dict(n=int(ok.sum()), MAE_rel=float(np.mean(np.abs(e - t))), bias_rel=float(np.mean(e - t)),
             MAE_vol=float(np.mean(np.abs(e - t) * rng_theta[ok] * 100)),
             corr=float(np.corrcoef(t, e)[0, 1]) if ok.sum() > 2 else np.nan)
    for thr in (0.4, 0.5, 0.6):
        dry_t, dry_e = t < thr, e < thr
        tp, fn = np.sum(dry_t & dry_e), np.sum(dry_t & ~dry_e)
        fp, tn = np.sum(~dry_t & dry_e), np.sum(~dry_t & ~dry_e)
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        tag = f"{int(thr * 100)}"
        r[f"acc_{tag}"] = float((tp + tn) / max(len(t), 1))
        r[f"recall_{tag}"] = float(rec)                  # частка справді сухих діб, які оцінка теж назвала сухими
        r[f"false_alarm_{tag}"] = float(fp / max(fp + tn, 1))
        r[f"bal_acc_{tag}"] = float((rec + spec) / 2)
    return r


def run():
    fr = pd.read_csv(io.PROCESSED / "stage3_france.csv.gz", parse_dates=["time"])
    meta = io.site_meta(io.FR_SITE)
    h = io.load_hourly(io.FR_SITE)
    h["time_naive"] = h["time"].dt.tz_localize(None) if h["time"].dt.tz is not None else h["time"]
    bundle = joblib.load(MODEL)
    rows, series = [], []
    for tg, cfg in TARGETS.items():
        model = bundle["models"][(cfg["core"], 1)]
        for st in b3.STATIONS:
            df = fr[fr["station"] == st].set_index("time").sort_index()
            e_w = df[f"e_{tg}"]
            # кліматологія ERA5-Land за роками навчання (згладжена ±7 діб)
            tr = df[df["year"].isin(b3.TRAIN_Y)]
            clim = tr.groupby("doy")[f"e_{tg}"].mean().reindex(range(1, 367)).interpolate(limit_direction="both")
            clim = clim.rolling(15, center=True, min_periods=1).mean()
            est = pd.DataFrame(index=df.index)
            est["truth"] = df[f"w_{tg}"]
            est["climatology"] = clim.reindex(df["doy"]).to_numpy()
            est["physics"] = physics_free(meta, b3.hwsd_soil(st), h, e_w, cfg["zr"]).reindex(df.index)
            est["era5l_now"] = e_w
            est["era5l_lag5_ml"] = rollout(df, f"e_{tg}", 5, model)
            est["sensor7_ml"] = rollout(df, f"w_{tg}", 7, model)
            est["sensor3_ml"] = rollout(df, f"w_{tg}", 3, model)
            te = est[est.index.year.isin(b3.TEST_Y)]
            rng = np.full(len(te), float(df[f"range_{tg}"].iloc[0]))
            for v in VARIANTS:
                rows.append(dict(target=tg, station=st, variant=v,
                                 **score(te["truth"].to_numpy(), te[v].to_numpy(), rng)))
            te = te.assign(station=st, target=tg)
            series.append(te.reset_index())
            print(f"  {tg} {st}: готово", flush=True)
    res = pd.DataFrame(rows)
    allrows = pd.concat(series, ignore_index=True)
    agg = []
    for tg in TARGETS:
        a = allrows[allrows.target == tg]
        rngs = np.array([float(fr[(fr.station == s)][f"range_{tg}"].iloc[0]) for s in a["station"]])
        for v in VARIANTS:
            agg.append(dict(target=tg, station="усі", variant=v, **score(a["truth"].to_numpy(), a[v].to_numpy(), rngs)))
    res = pd.concat([pd.DataFrame(agg), res], ignore_index=True)
    res.to_csv(RES / "no_sensor_metrics.csv", index=False)
    allrows.to_csv(RES / "no_sensor_series.csv.gz", index=False)
    report(res, allrows)
    return res


def report(res, allrows):
    # графік 1: точність рівня і рішення
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, title) in zip(axes, (("MAE_vol", "Похибка рівня вологості, % об."),
                                          ("corr", "Кореляція з датчиком"),
                                          ("bal_acc_50", "Збалансована точність «сухо/волого» (поріг 0.5)"))):
        x = np.arange(len(VARIANTS))
        for k, tg in enumerate(TARGETS):
            sub = res[(res.station == "усі") & (res.target == tg)].set_index("variant").loc[VARIANTS]
            ax.bar(x + (k - 0.5) * 0.38, sub[metric], 0.38, color=[COL[v] for v in VARIANTS],
                   alpha=0.55 if k == 0 else 1.0, label=TARGETS[tg]["uk"])
        ax.set_xticks(x, [UK[v].split(" (")[0].replace("ERA5-Land", "ERA5-L") for v in VARIANTS], rotation=35,
                      ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "bal_acc_50":
            ax.axhline(0.5, color="#888", lw=0.7, ls="--")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Точка на карті без датчика: оцінка поточного стану ґрунту (ISMN, тест 2018–2022)")
    fig.tight_layout()
    fig.savefig(RES / "fig4_no_sensor.png", dpi=150)
    plt.close(fig)
    # графік 2: сезон fraye 2019, 0–55 см
    s = allrows[(allrows.station == "fraye") & (allrows.target == "T3") & (allrows.time.dt.year == 2019)]
    if len(s):
        num = s.select_dtypes("number").columns
        s = s.set_index("time")[list(num)].asfreq("D").reset_index()       # розриви в даних — без прямих ліній
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.plot(s.time, s.truth, color="black", lw=1.6, label="датчик (правда)")
        for v in ("climatology", "physics", "era5l_lag5_ml", "sensor7_ml"):
            ax.plot(s.time, s[v], color=COL[v], lw=1.1, label=UK[v])
        ax.axhline(THR, color="#888", lw=0.7, ls="--")
        ax.text(s.time.iloc[0], THR + 0.02, "поріг «сухо»", fontsize=8, color="#666")
        ax.set_ylabel("відносна вологість")
        ax.set_title("fraye, профіль 0–55 см, сезон 2019: оцінка стану без датчика")
        ax.legend(fontsize=8, frameon=False, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(RES / "fig5_no_sensor_season.png", dpi=150)
        plt.close(fig)
    lines = ["# Точка на карті без датчика: оцінка поточного стану\n\n",
             "Відносні одиниці датчика (0 — найсухіше, 1 — найвологіше). «Сухо» — w < поріг. "
             "bal_acc — середнє з частки правильно впізнаних сухих і вологих діб (0.5 = вгадування).\n"]
    for tg in TARGETS:
        sub = res[(res.station == "усі") & (res.target == tg)]
        lines.append(f"\n## Профіль {TARGETS[tg]['uk']}\n\n| Варіант | MAE рівня, % об. | зміщення | кореляція | "
                     "точн. (0.5) | збаланс. точн. 0.4 / 0.5 / 0.6 | знайдено сухих діб (0.5) | хибні «сухо» (0.5) |\n"
                     "|---|---|---|---|---|---|---|---|\n")
        for _, r in sub.iterrows():
            lines.append(f"| {UK[r.variant]} | {r.MAE_vol:.2f} | {r.bias_rel:+.2f} | {r["corr"]:.2f} | {r.acc_50:.2f} | "
                         f"{r.bal_acc_40:.2f} / {r.bal_acc_50:.2f} / {r.bal_acc_60:.2f} | {r.recall_50:.2f} | "
                         f"{r.false_alarm_50:.2f} |\n")
        lines.append(f"\n**За станціями (MAE рівня, % об. / збаланс. точність 0.5), {TARGETS[tg]['uk']}**\n\n"
                     "| Варіант | fraye | grandcal | hillan2 |\n|---|---|---|---|\n")
        for v in VARIANTS:
            cells = []
            for st in b3.STATIONS:
                r = res[(res.station == st) & (res.target == tg) & (res.variant == v)].iloc[0]
                cells.append(f"{r.MAE_vol:.2f} / {r.bal_acc_50:.2f}")
            lines.append(f"| {UK[v]} | " + " | ".join(cells) + " |\n")
    (RES / "no_sensor_summary.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
