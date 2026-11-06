"""
Етап 3: моделі прогнозу зміни вологості на 1–2 доби проти РЕАЛЬНИХ датчиків ISMN (FR_Aqui).

Тест: 2018–2022 (сезони), станції fraye, grandcal, hillan2. Погода — факт ERA5 (архівних прогнозів
за ці роки немає), тож це верхня межа якості за погодою.
Моделі:
  persistence   M0  ŷ = 0
  era5land      R   зміна, яку дає сам ERA5-Land у цій клітинці (довідково: «що каже реаналіз»)
  physics       M1  фізика HydroBrain з ґрунтом з метаданих станції (HWSD), коренева зона 0.3 м
  physics_cal   M1c та сама фізика + лінійне калібрування масштабу на датчиках 2013–2017
  ml_ua         M2  бустинг, навчений лише на ERA5-Land 14 українських точок (перенесення без донавчання)
  ml_local      M5  бустинг, навчений на датчиках цих станцій 2013–2017
  ml_other      M5b бустинг, навчений на датчиках ІНШИХ станцій 2013–2017 (нова станція поруч)
  hybrid_local  M6  фізика + ML-поправка залишку, навчена на датчиках 2013–2017
Цілі: T1 — датчик 5 см; T2 — профіль 0–30 см; T3 — профіль 0–55 см.

Запуск: python -m ds.stage3.experiment  (після ds.stage3.build; ≈ 2–3 хв)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from ..stage2 import data_io as io  # noqa: E402
from ..stage2.experiment import monotone  # noqa: E402
from .build import STATIONS, TEST_Y, TRAIN_Y  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
MODELS = ["persistence", "era5land", "physics", "physics_cal", "phys_th_prior", "phys_th_soilcal",
          "phys_th_soilcal_lin", "ml_ua", "ml_ua_local", "ml_local", "ml_other", "hybrid_local", "hybrid_cal"]
MODEL_UK = dict(persistence="M0 Персистентність", era5land="R ERA5-Land (довідково)", physics="M1 Фізика",
                physics_cal="M1c Фізика + калібрування", ml_ua="M2 ML з України (ERA5-Land)",
                ml_local="M5 ML на датчиках", ml_other="M5b ML на сусідніх датчиках",
                hybrid_local="M6 Гібрид на датчиках", hybrid_cal="M6c Гібрид на калібр. фізиці",
                phys_th_prior="M1θ Фізика в θ, довідник", phys_th_soilcal="M1s Фізика в θ, θFC/θWP з датчика",
                phys_th_soilcal_lin="M1s+c Фізика в θ, θFC/θWP + калібрування",
                ml_ua_local="M2L ML з України + локальна поправка")
COL = dict(persistence="#9E9E9E", era5land="#B39DDB", physics="#C77C48", physics_cal="#E0A878",
           ml_ua="#5B7DB1", ml_local="#2E5C9A", ml_other="#8FA8D0", hybrid_local="#3A8F6B",
           hybrid_cal="#1F6B4A", phys_th_prior="#A0522D", phys_th_soilcal="#D2691E", phys_th_soilcal_lin="#F4A460", ml_ua_local="#3D6FB0")
PAST = ["P_past1", "P_past3", "P_past7", "ET0_past1", "ET0_past3", "ET0_past7", "T_past3"]
WX = ["P", "ET0", "Tmax", "RH", "SW", "U"]
FLUX = ["et", "pe", "dp", "ro"]
TARGETS = ("T1", "T2", "T3")
T_UK = dict(T1="Поверхня 5 см", T2="Профіль 0–30 см", T3="Профіль 0–55 см")
HYB_CAP = 0.3     # у відносних одиницях датчика; динаміка піску значно швидша, ніж 0–100 см в Україні
SEED = 0


def prep(df, t, h):
    x = df.copy()
    x["y"] = x[f"y_{t}_{h}"]
    x["w"] = x[f"w_{t}"]
    x["dw1"] = x[f"dw_{t}_past1"]
    x["dw3"] = x[f"dw_{t}_past3"]
    x["doy_sin"] = np.sin(2 * np.pi * x["doy"] / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * x["doy"] / 365.25)
    x["P_h"] = sum(x[f"P_{j}"] for j in range(1, h + 1))
    x["ET0_h"] = sum(x[f"ET0_{j}"] for j in range(1, h + 1))
    if f"phys_{t}_{h}" in x:
        x["phys"] = x[f"phys_{t}_{h}"]
        for k in FLUX:
            x[f"ph_{k}"] = sum(x[f"{k}_{j}"] for j in range(1, h + 1))
        x["era5"] = x[f"y_E{t}_{h}"]
        x["thp"] = x[f"physthp_{t}_{h}"]
        x["thc"] = x[f"physthc_{t}_{h}"]
        x["rng"] = x[f"range_{t}"]
    return x


def feats(h, phys=False):
    f = ["w", "dw1", "dw3"] + PAST + ["doy_sin", "doy_cos"] + [f"{v}_{j}" for j in range(1, h + 1) for v in WX]
    f += ["P_h", "ET0_h"]
    if phys:
        f += ["phys"] + [f"ph_{k}" for k in FLUX]
    return f


def gbm(cols, big, mono=True):
    kw = dict(learning_rate=0.05, loss="absolute_error", random_state=SEED, l2_regularization=1.0,
              max_iter=600 if big else 300, max_leaf_nodes=31 if big else 15, min_samples_leaf=40 if big else 20)
    if mono:
        kw["monotonic_cst"] = monotone(cols)
    return HistGradientBoostingRegressor(**kw)


def metrics(d, pred):
    k = d["rng"].to_numpy() * 100
    e = (np.asarray(pred) - d["y"].to_numpy()) * k
    y = d["y"].to_numpy() * k
    p = d["P_h"].to_numpy()
    wet, dry = p >= 5, p < 0.5
    return dict(n=len(d), MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
                MAE_rain=float(np.mean(np.abs(e[wet]))), MAE_dry=float(np.mean(np.abs(e[dry]))),
                bias=float(np.mean(e)))


def run():
    RES.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(io.PROCESSED / "stage3_france.csv.gz", parse_dates=["time"])
    ua = pd.read_csv(io.PROCESSED / "stage3_ua.csv.gz", parse_dates=["time"])
    rows, store, extras = [], [], {}
    for t in TARGETS:
        for h in (1, 2):
            f = prep(fr, t, h)
            u = prep(ua, t, h).dropna(subset=["y"])
            need = ["y", "w", "phys", "era5", "thp", "thc"]
            tr = f[f["year"].isin(TRAIN_Y)].dropna(subset=need)
            te = f[f["year"].isin(TEST_Y)].dropna(subset=need)
            c0, c1 = feats(h), feats(h, True)
            P = {"persistence": np.zeros(len(te)), "era5land": te["era5"].to_numpy(),
                 "physics": te["phys"].to_numpy()}
            a, b = np.polyfit(tr["phys"], tr["y"], 1)
            extras[f"physics_cal_{t}_h{h}"] = dict(slope=float(a), intercept=float(b))
            P["physics_cal"] = a * te["phys"].to_numpy() + b
            P["phys_th_prior"] = te["thp"].to_numpy()
            P["phys_th_soilcal"] = te["thc"].to_numpy()
            a2, b2 = np.polyfit(tr["thc"], tr["y"], 1)
            extras[f"phys_th_soilcal_lin_{t}_h{h}"] = dict(slope=float(a2), intercept=float(b2))
            P["phys_th_soilcal_lin"] = a2 * te["thc"].to_numpy() + b2
            m_ua = gbm(c0, True).fit(u[c0], u["y"])
            P["ml_ua"] = m_ua.predict(te[c0])
            # M2L: ядро з України + локальна поправка залишку на датчиках 2013–2017 (стратегія HydroBrain 1.6)
            trc, tec = tr.copy(), te.copy()
            trc["core"], tec["core"] = m_ua.predict(tr[c0]), P["ml_ua"]
            ml = gbm(c0 + ["core"], False, mono=False).fit(trc[c0 + ["core"]], trc["y"] - trc["core"])
            P["ml_ua_local"] = P["ml_ua"] + ml.predict(tec[c0 + ["core"]])
            P["ml_local"] = gbm(c0, False).fit(tr[c0], tr["y"]).predict(te[c0])
            mo = np.full(len(te), np.nan)
            for st in STATIONS:
                m = gbm(c0, False).fit(tr[tr["station"] != st][c0], tr[tr["station"] != st]["y"])
                sel = (te["station"] == st).to_numpy()
                mo[sel] = m.predict(te[sel][c0])
            P["ml_other"] = mo
            hm = gbm(c1, False, mono=False).fit(tr[c1], tr["y"] - tr["phys"])
            P["hybrid_local"] = te["phys"].to_numpy() + np.clip(hm.predict(te[c1]), -HYB_CAP, HYB_CAP)
            # M6c: база — КАЛІБРОВАНА фізика (сира фізика тут реагує в ~5 разів сильніше за датчик,
            # і такий залишок ML не може надійно виправити)
            base_tr = a * tr["phys"] + b
            hc = gbm(c1, False, mono=False).fit(tr[c1], tr["y"] - base_tr)
            P["hybrid_cal"] = P["physics_cal"] + np.clip(hc.predict(te[c1]), -HYB_CAP, HYB_CAP)
            for m in MODELS:
                rows.append(dict(target=t, h=h, station="усі", model=m, **metrics(te, P[m])))
                for st in STATIONS:
                    sel = (te["station"] == st).to_numpy()
                    rows.append(dict(target=t, h=h, station=st, model=m, **metrics(te[sel], P[m][sel])))
            s = te[["station", "time", "y", "rng", "P_h", "w"]].copy()
            for m in MODELS:
                s[m] = P[m]
            s["target"], s["h"] = t, h
            store.append(s)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "metrics.csv", index=False)
    pd.concat(store).to_csv(RES / "predictions_test.csv.gz", index=False)
    (RES / "extras.json").write_text(json.dumps(extras, indent=1), encoding="utf-8")
    report(res, extras)
    return res


def report(res, extras):
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    for i, t in enumerate(TARGETS):
        for j, h in enumerate((1, 2)):
            ax = axes[i, j]
            sub = res[(res.target == t) & (res.h == h) & (res.station == "усі")].set_index("model").loc[MODELS]
            x = np.arange(len(MODELS))
            ax.bar(x, sub["MAE"], color=[COL[m] for m in MODELS])
            for k, v in enumerate(sub["MAE"]):
                ax.text(k, min(v, ax.get_ylim()[1] if False else v), f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x, [MODEL_UK[m].split(" ", 1)[0] for m in MODELS], fontsize=7, rotation=45)
            ax.set_title(T_UK[t] + f", горизонт {h} доб.")
            ax.set_ylabel("MAE, % об.")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Реальні датчики ISMN (FR_Aqui), тест 2018–2022, погода — факт ERA5")
    fig.tight_layout()
    fig.savefig(RES / "fig1_mae_sensors.png", dpi=150)
    plt.close(fig)
    p = pd.read_csv(RES / "predictions_test.csv.gz", parse_dates=["time"])
    s = p[(p.station == "fraye") & (p.target == "T2") & (p.h == 1) & (p.time.dt.year == 2019)
          & (p.time.dt.month.between(4, 9))].sort_values("time")
    if len(s):
        k = s["rng"].iloc[0] * 100
        fig, ax = plt.subplots(figsize=(12, 3.6))
        ax.plot(s.time, s.y * k, color="black", lw=1.2, label="датчик (правда)")
        for m in ("physics_cal", "ml_ua", "ml_local", "hybrid_local"):
            ax.plot(s.time, s[m] * k, color=COL[m], lw=1, label=MODEL_UK[m])
        ax.axhline(0, color="#BBBBBB", lw=0.5)
        ax.set_ylabel("зміна за добу, % об.")
        ax.set_title("fraye, профіль 0–30 см, квітень–вересень 2019: прогноз на 1 добу")
        ax.legend(fontsize=8, frameon=False, ncol=5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(RES / "fig2_example_fraye_2019.png", dpi=150)
        plt.close(fig)
    lines = ["# Етап 3 — реальні датчики ISMN (FR_Aqui)\n",
             "MAE/RMSE — % об.; тест 2018–2022; MAE_rain — доби з опадами ≥ 5 мм на горизонті.\n"]
    for t in TARGETS:
        for h in (1, 2):
            sub = res[(res.target == t) & (res.h == h) & (res.station == "усі")]
            lines.append(f"\n## {T_UK[t]}, горизонт {h} доб.\n\n")
            lines.append("| Модель | MAE | RMSE | R² | MAE_rain | MAE_dry | bias |\n|---|---|---|---|---|---|---|\n")
            for _, r in sub.iterrows():
                lines.append(f"| {MODEL_UK[r.model]} | {r.MAE:.3f} | {r.RMSE:.3f} | {r.R2:.3f} | {r.MAE_rain:.3f} | "
                             f"{r.MAE_dry:.3f} | {r.bias:+.3f} |\n")
    for t in ("T2", "T3"):
        for val, nm in (("MAE", "MAE"), ("bias", "зміщення (bias)")):
            lines.append(f"\n## За станціями: {nm}, {T_UK[t]}, 1 доба\n\n")
            sub = res[(res.target == t) & (res.h == 1) & (res.station != "усі")]
            piv = sub.pivot(index="model", columns="station", values=val).loc[MODELS]
            lines.append("| Модель | " + " | ".join(piv.columns) + " |\n" + "|---" * (len(piv.columns) + 1) + "|\n")
            for m, r in piv.iterrows():
                lines.append(f"| {MODEL_UK[m]} | " + " | ".join(f"{v:+.3f}" if val == "bias" else f"{v:.3f}" for v in r)
                             + " |\n")
    lines.append("\n```json\n" + json.dumps(extras, indent=1) + "\n```\n")
    (RES / "summary.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
