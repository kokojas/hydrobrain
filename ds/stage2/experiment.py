"""
Етап 2: порівняння моделей прогнозу зміни вологості кореневої зони (0–100 см) на 1 і 2 доби.

Ціль: y_h = w(d+h) − w(d), w — відносна вологість ERA5-Land (див. data_io.py).
Моделі:
  M0  persistence      — «нічого не зміниться»: ŷ = 0
  M1  physics          — фізика HydroBrain (FAO-56), старт від спостереженого стану
  M1c physics_cal      — фізика з лінійним калібруванням масштабу для кожної точки (на тренувальних роках)
  M2  ml_only          — градієнтний бустинг лише на стані ґрунту і погоді (без фізики)
  M3  ml_physfeat      — той самий бустинг + виходи фізики як ознаки
  M4  hybrid           — фізика + ML-поправка: бустинг вчить залишок y − ŷ_фізики (поправка обмежена)
Розбиття: train 2016–2021, val 2022 (вибір гіперпараметрів), test 2023–2025.
Погода на горизонт: ideal (факт ERA5) або fcst (справжній архівний прогноз, лише 2024–2025).
Моделі навчаються на ideal; у режимі fcst на вхід подається прогноз — як було б у роботі.

Запуск:  python -m ds.stage2.experiment      (після build_dataset; ≈ 3–5 хв)
Результати: ds/stage2/results/*.csv, *.png, summary.md
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import data_io as io

RES = Path(__file__).resolve().parent / "results"
HORIZONS = (1, 2)
MODELS = ["persistence", "physics", "physics_cal", "ml_only", "ml_physfeat", "hybrid"]
MODEL_UK = dict(persistence="M0 Персистентність", physics="M1 Фізика", physics_cal="M1c Фізика + калібрування",
                ml_only="M2 Лише ML", ml_physfeat="M3 ML + ознаки фізики", hybrid="M4 Гібрид (фізика + ML)")

STATE = ["w_rz", "w__0_7", "w_7_28", "w__100", "dw_past1", "dw_past3"]
PAST = ["P_past1", "P_past3", "P_past7", "ET0_past1", "ET0_past3", "ET0_past7", "T_past3"]
SEAS = ["doy_sin", "doy_cos"]
STATIC = ["theta_range", "elevation"]
WX_H = ["P", "ET0", "Tmax", "RH", "SW", "U"]
PHYS_FLUX = ["et", "e", "t", "pe", "dp", "ro"]
HYBRID_CAP = 0.06          # макс. поправка ML у відносних одиницях (≈ 1 % об. при типовому діапазоні 0.18)
SEED = 0


# ---------------------------------------------------------------------------- дані
def load(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or io.PROCESSED / "daily.csv.gz", parse_dates=["time"])
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    return df


def view(df: pd.DataFrame, h: int, mode: str) -> pd.DataFrame:
    """Таблиця для горизонту h і режиму погоди: погода на доби 1..h і фізика — з потрібного режиму."""
    out = df[["site", "time", "year", "theta_range", f"y{h}"] + STATE + PAST + SEAS + ["elevation"]].copy()
    out = out.rename(columns={f"y{h}": "y"})
    for j in range(1, h + 1):
        for v in WX_H:
            out[f"{v}_{j}"] = df[f"{v}_{j}_{mode}"] if f"{v}_{j}_{mode}" in df else np.nan
    out["P_h"] = sum(out[f"P_{j}"] for j in range(1, h + 1))
    out["P_act"] = sum(df[f"P_{j}_ideal"] for j in range(1, h + 1))      # фактичні опади (для категорій подій)
    out["ET0_h"] = sum(out[f"ET0_{j}"] for j in range(1, h + 1))
    ph = f"wphys_{h}_{mode}"
    out["phys"] = (df[ph] - df["w_rz"]) if ph in df else np.nan
    for k in PHYS_FLUX:
        cols = [f"{k}_{j}_{mode}" for j in range(1, h + 1)]
        out[f"ph_{k}"] = df[cols].sum(axis=1, min_count=len(cols)) if all(c in df for c in cols) else np.nan
    return out


def feats(h: int, with_phys: bool):
    f = STATE + PAST + SEAS + STATIC
    f += [f"{v}_{j}" for j in range(1, h + 1) for v in WX_H] + ["P_h", "ET0_h"]
    if with_phys:
        f += ["phys"] + [f"ph_{k}" for k in PHYS_FLUX]
    return f


def monotone(cols):
    """Фізична узгодженість для M2/M3: більше опадів на горизонті → не сухіше; більша ETo → не вологіше."""
    m = []
    for c in cols:
        if c.startswith("P_") and c[2:] in ("1", "2", "h"):
            m.append(1)
        elif c.startswith("ET0_") and c[4:] in ("1", "2", "h"):
            m.append(-1)
        else:
            m.append(0)
    return m


# ---------------------------------------------------------------------------- моделі
GRID = dict(max_iter=[300, 600], max_leaf_nodes=[15, 31], loss=["squared_error", "absolute_error"])


def gbm(params, cols, mono=True, loss=None, quantile=None):
    kw = dict(learning_rate=0.05, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
              max_iter=params["max_iter"], max_leaf_nodes=params["max_leaf_nodes"],
              loss=loss or params["loss"])
    if quantile is not None:
        kw.update(loss="quantile", quantile=quantile)
    if mono:
        kw["monotonic_cst"] = monotone(cols)
    return HistGradientBoostingRegressor(**kw)


def fit_physics_cal(tr: pd.DataFrame):
    """y ≈ a·phys + c для кожної точки (+ спільна для нових точок)."""
    cal = {}
    for s, g in tr.groupby("site"):
        g = g.dropna(subset=["phys", "y"])
        a, c = np.polyfit(g["phys"], g["y"], 1)
        cal[s] = (float(a), float(c))
    g = tr.dropna(subset=["phys", "y"])
    cal["_pooled"] = tuple(map(float, np.polyfit(g["phys"], g["y"], 1)))
    return cal


def apply_cal(df, cal):
    ac = np.array([cal.get(s, cal["_pooled"]) for s in df["site"]])
    return ac[:, 0] * df["phys"].to_numpy() + ac[:, 1]


class Suite:
    """Навчає всі моделі для одного горизонту на тренувальній вибірці (режим ideal)."""

    def __init__(self, h, params=None):
        self.h = h
        self.params = params or {}

    def fit(self, tr: pd.DataFrame, va: pd.DataFrame | None = None):
        h = self.h
        self.cal = fit_physics_cal(tr)
        self.cols = {"ml_only": feats(h, False), "ml_physfeat": feats(h, True), "hybrid": feats(h, True)}
        if va is not None and not self.params:
            self.params = {m: self._tune(m, tr, va) for m in ("ml_only", "ml_physfeat", "hybrid")}
        self.models = {}
        for m in ("ml_only", "ml_physfeat"):
            self.models[m] = gbm(self.params[m], self.cols[m]).fit(tr[self.cols[m]], tr["y"])
        r = tr["y"] - tr["phys"]
        self.models["hybrid"] = gbm(self.params["hybrid"], self.cols["hybrid"], mono=False).fit(tr[self.cols["hybrid"]], r)
        self.q = {q: gbm(self.params["hybrid"], self.cols["hybrid"], mono=False, quantile=q)
                  .fit(tr[self.cols["hybrid"]], r) for q in (0.1, 0.9)}
        return self

    def _tune(self, m, tr, va):
        best, best_p = 1e9, None
        for mi, ml, lo in itertools.product(GRID["max_iter"], GRID["max_leaf_nodes"], GRID["loss"]):
            p = dict(max_iter=mi, max_leaf_nodes=ml, loss=lo)
            cols = self.cols[m]
            if m == "hybrid":
                mdl = gbm(p, cols, mono=False).fit(tr[cols], tr["y"] - tr["phys"])
                pred = va["phys"] + np.clip(mdl.predict(va[cols]), -HYBRID_CAP, HYBRID_CAP)
            else:
                pred = gbm(p, cols).fit(tr[cols], tr["y"]).predict(va[cols])
            e = float(np.mean(np.abs(pred - va["y"])))
            if e < best:
                best, best_p = e, p
        return best_p

    def predict(self, df: pd.DataFrame) -> dict:
        P = {"persistence": np.zeros(len(df)), "physics": df["phys"].to_numpy(),
             "physics_cal": apply_cal(df, self.cal)}
        for m in ("ml_only", "ml_physfeat"):
            P[m] = self.models[m].predict(df[self.cols[m]])
        corr = np.clip(self.models["hybrid"].predict(df[self.cols["hybrid"]]), -HYBRID_CAP, HYBRID_CAP)
        P["hybrid"] = df["phys"].to_numpy() + corr
        self.last_q = {q: df["phys"].to_numpy() + np.clip(mq.predict(df[self.cols["hybrid"]]), -HYBRID_CAP * 2,
                                                         HYBRID_CAP * 2) for q, mq in self.q.items()}
        return P


# ---------------------------------------------------------------------------- метрики
def metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """MAE/RMSE у % об. (відносні одиниці × діапазон θ точки × 100), R², помилки на подіях."""
    k = df["theta_range"].to_numpy() * 100.0
    e = (pred - df["y"].to_numpy()) * k
    y = df["y"].to_numpy() * k
    ok = np.isfinite(e)
    e, y = e[ok], y[ok]
    ph = df["P_act"].to_numpy()[ok]
    wet = ph >= 5.0
    dry = ph < 0.5
    return dict(n=int(ok.sum()), MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
                MAE_rain=float(np.mean(np.abs(e[wet]))) if wet.any() else np.nan,
                MAE_dry=float(np.mean(np.abs(e[dry]))) if dry.any() else np.nan,
                bias=float(np.mean(e)))


def table(df, P, extra=None):
    rows = []
    for m in MODELS:
        r = dict(model=m, **metrics(df, P[m]))
        if extra:
            r.update(extra)
        rows.append(r)
    return pd.DataFrame(rows)


def monotonicity_violations(suite: Suite, df: pd.DataFrame, add_mm=5.0) -> dict:
    """Частка прогнозів, які стають СУХІШИМИ, якщо додати add_mm дощу в першу добу горизонту.
    ОБМЕЖЕННЯ: ознаки фізики тут не перераховуються (фізику не перезапускаємо з додатковим дощем).
    Для M2 це повна перевірка. Для M3 — перевірка при «замороженій» фізиці. Для M4 (гібрид) число
    НЕ означає, що гібрид порушує фізику: перевіряється лише ML-поправка окремо, а в реальному прогоні
    фізика сама додала б вологу від дощу. Повна перевірка гібрида потребує перезапуску фізики."""
    d2 = df.copy()
    d2["P_1"] = d2["P_1"] + add_mm
    d2["P_h"] = d2["P_h"] + add_mm
    a, b = suite.predict(df), suite.predict(d2)
    return {m: float(np.mean(b[m] < a[m] - 1e-6)) for m in ("ml_only", "ml_physfeat", "hybrid")}


# ---------------------------------------------------------------------------- експерименти
def run(df: pd.DataFrame, verbose=True):
    RES.mkdir(parents=True, exist_ok=True)
    ua = df[df["site"].isin(io.UA_SITES)]
    fr = df[df["site"] == io.FR_SITE]
    results, extras = [], {}
    preds_store = []
    for h in HORIZONS:
        t0 = time.time()
        vi = view(ua, h, "ideal")
        tr = vi[vi["year"].isin(io.TRAIN_YEARS)].dropna(subset=["y", "phys"])
        va = vi[vi["year"].isin(io.VAL_YEARS)].dropna(subset=["y", "phys"])
        te = vi[vi["year"].isin(io.TEST_YEARS)].dropna(subset=["y", "phys"])
        suite = Suite(h).fit(tr, va)
        extras[f"params_h{h}"] = suite.params
        extras[f"physics_cal_pooled_h{h}"] = suite.cal["_pooled"]
        # (A) часовий тест, ideal, 2023–2025
        P = suite.predict(te)
        results.append(table(te, P, dict(h=h, split="A_test_2023-25", weather="ideal")))
        q10, q90 = suite.last_q[0.1], suite.last_q[0.9]
        cov = float(np.mean((te["y"] >= q10) & (te["y"] <= q90)))
        extras[f"hybrid_q10_q90_coverage_h{h}"] = cov
        extras[f"monotonicity_violations_h{h}"] = monotonicity_violations(suite, te)
        store = te[["site", "time", "y", "theta_range", "P_act", "w_rz"]].copy()
        for m in MODELS:
            store[m] = P[m]
        store["h"] = h
        store["weather"] = "ideal"
        preds_store.append(store)
        # (B) 2024–2025: ідеальна погода vs справжній прогноз на ТИХ САМИХ добах
        vf = view(ua, h, "fcst")
        vf = vf[vf["year"].isin(io.FORECAST_YEARS)].dropna(subset=["y", "phys"])
        vi_same = vi.set_index(["site", "time"]).loc[vf.set_index(["site", "time"]).index].reset_index()
        Pi, Pf = suite.predict(vi_same), suite.predict(vf)
        results.append(table(vi_same, Pi, dict(h=h, split="B_2024-25", weather="ideal")))
        results.append(table(vf, Pf, dict(h=h, split="B_2024-25", weather="fcst")))
        s2 = vf[["site", "time", "y", "theta_range", "P_act", "w_rz"]].copy()
        for m in MODELS:
            s2[m] = Pf[m]
        s2["h"] = h
        s2["weather"] = "fcst"
        preds_store.append(s2)
        # (C) leave-location-out: нова точка, тестові роки
        lolo = []
        for s in io.UA_SITES:
            tr_s = tr[tr["site"] != s]
            te_s = te[te["site"] == s]
            st = Suite(h, params=suite.params).fit(tr_s)
            Ps = st.predict(te_s)
            t = table(te_s, Ps, dict(h=h, split="C_LOLO", weather="ideal", site=s))
            lolo.append(t)
        lolo = pd.concat(lolo)
        lolo.to_csv(RES / f"lolo_by_site_h{h}.csv", index=False)
        agg = lolo.groupby("model")[["MAE", "RMSE", "R2", "MAE_rain", "MAE_dry", "bias"]].mean().reset_index()
        agg["n"] = lolo.groupby("model")["n"].sum().values
        agg["h"], agg["split"], agg["weather"] = h, "C_LOLO", "ideal"
        results.append(agg)
        # (D) перенесення на інший клімат: Франція (ERA5-Land), модель навчена на Україні
        if len(fr):
            vfr = view(fr, h, "ideal").dropna(subset=["y", "phys"])
            Pfr = suite.predict(vfr)
            results.append(table(vfr, Pfr, dict(h=h, split="D_France", weather="ideal")))
        if verbose:
            print(f"h={h}: готово за {time.time() - t0:.0f} с", flush=True)
    res = pd.concat(results, ignore_index=True)
    res.to_csv(RES / "metrics.csv", index=False)
    pd.concat(preds_store).to_csv(RES / "predictions_test.csv.gz", index=False)
    (RES / "extras.json").write_text(json.dumps(extras, indent=1, ensure_ascii=False), encoding="utf-8")
    return res, extras


def main():
    df = load()
    res, extras = run(df)
    from . import report
    report.make(res, extras)


if __name__ == "__main__":
    main()
