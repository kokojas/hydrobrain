"""
Навчання «ядра прогнозу» HydroBrain: ML-моделі зміни вологості на 24 і 48 год, навчені на ERA5-Land
14 українських точок (2016–2025). Обґрунтування вибору — Етапи 2–3 (див. CHANGELOG 1.2–1.5).

Чотири цілі за глибиною шару (вибирається в HydroBrain за глибиною коренів):
  T1 — 0–7 см,  T2 — 0–30 см,  T3 — 0–55 см,  RZ — 0–100 см.
Усе у відносних одиницях (0 ≈ найсухіше, 1 ≈ найвологіше), тож на полі потрібен лише масштаб датчика.

Процедура:
  1) оцінка якості «як у роботі»: навчання на 2016–2023 з фактичною погодою, перевірка на 2024–2025
     зі СПРАВЖНІМ архівним прогнозом погоди → MAE і σ похибки (σ передається в Монте-Карло рішення);
  2) фінальні моделі — на всіх роках 2016–2025.
Результат: hydrobrain/models/ml_core_v2.joblib (+ ml_core_v2.json з метаданими і якістю).
1.10 (v2): профілі 0–30 і 0–55 см рахуються узгоджено із застосунком — спершу агрегування шарів у м³/м³,
потім нормалізація (у v1 усереднювались уже нормалізовані шари).

Запуск: python -m ds.core.train_core   (≈ 10 хв)
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from ..stage2 import data_io as io
from ..stage3.build import add_targets
from ..stage3.experiment import feats, gbm, prep

OUT = io.ROOT / "hydrobrain" / "models"
TARGETS = dict(T1=dict(depth_cm=(0, 7)), T2=dict(depth_cm=(0, 30)), T3=dict(depth_cm=(0, 55)),
               RZ=dict(depth_cm=(0, 100)))
WX = ["P", "ET0", "Tmax", "RH", "SW", "U"]
PAST = ["P_past1", "P_past3", "P_past7", "ET0_past1", "ET0_past3", "ET0_past7", "T_past3"]


def _with_profiles(d: pd.DataFrame) -> pd.DataFrame:
    """Додає w_T2, w_T3 (відносна вологість профілів, узгоджено із застосунком) з сирих рядів ERA5-Land."""
    parts = []
    for s in d["site"].unique():
        lp = io.layer_profiles(s).reset_index().rename(columns={"index": "time"})
        lp["site"] = s
        parts.append(lp)
    lp = pd.concat(parts, ignore_index=True)
    lp["time"] = pd.to_datetime(lp["time"])
    return d.drop(columns=[c for c in ("w_T2", "w_T3") if c in d]).merge(lp, on=["site", "time"], how="left")


def build():
    d = pd.read_csv(io.PROCESSED / "daily.csv.gz", parse_dates=["time"])
    d = d[d["site"].isin(io.UA_SITES)].copy()
    d["w_T1"] = d["w__0_7"]
    d = _with_profiles(d)                         # 1.10: T2/T3 — агрегування в м³/м³, потім нормалізація
    d["w_RZ"] = d["w_rz"]
    out = []
    for (s, y), g in d.groupby(["site", "year"]):
        g = g.set_index("time").sort_index()
        for t in TARGETS:
            g = add_targets(g, f"w_{t}", t)
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def view(df, mode):
    x = df.copy()
    for j in (1, 2):
        for v in WX:
            c = f"{v}_{j}_{mode}"
            x[f"{v}_{j}"] = x[c] if c in x else np.nan
    return x


def main():
    t0 = time.time()
    df = build()
    ideal, fcst = view(df, "ideal"), view(df, "fcst")
    models, quality, evaluation_models = {}, {}, {}
    for t in TARGETS:
        for h in (1, 2):
            cols = feats(h)
            tr = prep(ideal[ideal["year"] <= 2023], t, h).dropna(subset=["y"])
            te = prep(fcst[fcst["year"].isin([2024, 2025])], t, h).dropna(subset=["y", "P_1"])
            m = gbm(cols, True).fit(tr[cols], tr["y"])
            evaluation_models[(t, h)] = m
            r = te["y"].to_numpy() - m.predict(te[cols])
            quality[f"{t}_h{h}"] = dict(mae_rel=float(np.mean(np.abs(r))), sigma_rel=float(np.std(r)),
                                        persistence_mae_rel=float(np.mean(np.abs(te["y"]))), n_test=int(len(te)))
            full = prep(ideal, t, h).dropna(subset=["y"])
            models[(t, h)] = gbm(cols, True).fit(full[cols], full["y"])
            print(f"  {t} h{h}: MAE {quality[f'{t}_h{h}']['mae_rel']:.4f} (персистентність "
                  f"{quality[f'{t}_h{h}']['persistence_mae_rel']:.4f}), {time.time() - t0:.0f} с", flush=True)
    meta = dict(version="ml_core_v2", sklearn_version=sklearn.__version__, trained_on="ERA5-Land, 14 точок України, 2016–2025",
                features={str(h): feats(h) for h in (1, 2)}, targets=TARGETS, quality_forecast_2024_2025=quality,
                units="відносні: Δw = Δθ / (θ_high − θ_low)",
                note="Якість оцінена на ERA5-Land із справжнім прогнозом погоди; на реальних датчиках похибка більша (Етап 3).")
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation_meta = dict(meta, training_end_year=2023, trained_on="ERA5-Land 2016–2023")
    joblib.dump(dict(models=evaluation_models, meta=evaluation_meta), OUT / "ml_core_evaluation.joblib", compress=3)
    joblib.dump(dict(models=models, meta=meta), OUT / "ml_core_v2.joblib", compress=3)
    (OUT / "ml_core_v2.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Збережено за {time.time() - t0:.0f} с: {OUT / 'ml_core_v2.joblib'}")


if __name__ == "__main__":
    main()
