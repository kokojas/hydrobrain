"""
Ядро прогнозу природної (без поливу) зміни вологості — ML, навчена на ERA5-Land України.

Чому так (дані — CHANGELOG 1.2–1.5): на українському ERA5-Land ML найточніша; на реальних датчиках
серед «готових» моделей вона переноситься найкраще, а на новому полі з масштабом від датчика вже за
1–4 тижні стабільно краща за персистентність. Фізика лишається для ефекту поливу і розкиду сценаріїв.

Як працює в HydroBrain:
  1. Масштаб датчика (θ_low, θ_high): 1-й/99-й перцентилі показів, якщо є ≥ 7 діб даних (розширений до
     ≥ 3 % об., як у перевірці); інакше — довідникові θWP/θFC поля.
  2. Ознаки: відносна вологість зараз і її зміна за 1 і 3 доби, погода за 1/3/7 діб у минулому і на
     кожну добу горизонту (прогноз), сезон.
  3. Прогноз Δw на 24 і 48 год → Δθ = Δw·(θ_high − θ_low) → зміна дефіциту кореневої зони
     ΔDr = −Δθ·1000·Zr (те саме припущення, що й у фільтра Калмана: датчик представляє кореневу зону).
  4. Рішення: траєкторії фізики у Монте-Карло зсуваються так, щоб на 24 і 48 год збігтися з прогнозом ML;
     σ похибки ML (оцінена на справжньому прогнозі погоди 2024–2025) додається як розкид сценаріїв.
  5. Після сезону даних поля — локальна ML-поправка до ядра (LocalResidual), вмикається лише якщо на
     відкладених даних поля вона справді зменшує похибку.
"""
from __future__ import annotations

import math
import os
from datetime import timedelta

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ml_core_v2.joblib")
MIN_RANGE = 0.03
# Похибки оцінки поточного стану БЕЗ свіжого датчика (Етап 3, ISMN, ds/stage3/no_sensor.py), відносні одиниці:
#   ERA5-Land −5 діб + ML: MAE ≈ 0.21–0.23 → σ ≈ 1.25·MAE;  датчик L діб тому + ML: MAE ≈ 0.024 + 0.0087·L
SIGMA_REANALYSIS = 0.28
STALE_MAX_DAYS = 10            # старий показ датчика прокручуємо не далі (перевірено до 7 діб)


# Домен навчання ML-ядра (1.9): 14 сільськогосподарських точок України (ds/fetch_data.py, LOCATIONS)
TRAIN_POINTS = [(46.75, 33.30), (47.20, 31.80), (46.40, 30.20), (47.60, 35.60), (48.30, 34.60), (48.40, 32.60),
                (49.50, 34.20), (49.70, 36.60), (49.20, 31.70), (49.10, 28.70), (50.20, 30.90), (50.40, 28.30),
                (49.40, 25.80), (50.80, 34.40)]
UA_BOX = (44.0, 52.5, 22.0, 40.5)      # широта min/max, довгота min/max — грубі межі України
FAR_KM = 300.0                         # далі від усіх точок навчання — позначка «далеко від даних навчання»
LOCAL_MIN_ROWS = 60                    # стільки діб із мітками потрібно для локального донавчання
LOCAL_RETRY_DAYS = 30                  # повторна спроба донавчання, якщо попередня не покращила прогноз


def in_domain(lat: float, lon: float):
    """(у межах України?, відстань до найближчої точки навчання, км)."""
    r = 6371.0
    def dist(a, b):
        p1, p2 = math.radians(a[0]), math.radians(b[0])
        dp, dl = p2 - p1, math.radians(b[1] - a[1])
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(h))
    near = min(dist((lat, lon), p) for p in TRAIN_POINTS)
    inside = UA_BOX[0] <= lat <= UA_BOX[1] and UA_BOX[2] <= lon <= UA_BOX[3]
    return inside, float(near)


def sigma_stale(lag_days: float) -> float:
    return float(min(1.25 * (0.024 + 0.0087 * lag_days), SIGMA_REANALYSIS))
MIN_SCALE_DAYS = 7
WX = ["P", "ET0", "Tmax", "RH", "SW", "U"]
PAST = ["P_past1", "P_past3", "P_past7", "ET0_past1", "ET0_past3", "ET0_past7", "T_past3"]


def feature_names(h: int):
    f = ["w", "dw1", "dw3"] + PAST + ["doy_sin", "doy_cos"] + [f"{v}_{j}" for j in range(1, h + 1) for v in WX]
    return f + ["P_h", "ET0_h"]


def choose_target(root_depth_m: float) -> str:
    """Шар ERA5-Land, на якому вчилась модель, за глибиною коренів."""
    if root_depth_m <= 0.4:
        return "T2"
    if root_depth_m <= 0.75:
        return "T3"
    return "RZ"


def sensor_scale(times, thetas, prior_low, prior_high):
    """Масштаб датчика: (θ_low, θ_high, джерело)."""
    th = np.asarray(thetas, float)
    th = th[np.isfinite(th)]
    if len(th) >= MIN_SCALE_DAYS and times:
        span_days = (max(times) - min(times)).total_seconds() / 86400.0
        if span_days >= MIN_SCALE_DAYS:
            lo, hi = float(np.percentile(th, 1)), float(np.percentile(th, 99))
            if hi - lo < MIN_RANGE:
                c = (hi + lo) / 2
                lo, hi = c - MIN_RANGE / 2, c + MIN_RANGE / 2
            return lo, hi, f"датчик ({span_days:.0f} діб)"
    return float(prior_low), float(max(prior_high, prior_low + MIN_RANGE)), "довідник (замало даних датчика)"


def daily_weather(times, now, precip, eto, temp, rh, sw, wind):
    """Агрегати погоди відносно моменту now: минуле (24/72/168 год до now) і доби горизонту після now."""
    t = np.array([(x - now).total_seconds() / 3600.0 for x in times])   # год відносно now (мітка = кінець години)
    arr = lambda a: np.nan_to_num(np.asarray(a, float)) if a is not None else np.zeros(len(t))
    P, E = arr(precip), arr(eto)
    out = {}
    for k, hrs in (("1", 24), ("3", 72), ("7", 168)):
        m = (t > -hrs) & (t <= 0)
        full = m.sum() >= hrs - 2
        out["P_past" + k] = float(P[m].sum()) if full else np.nan
        out["ET0_past" + k] = float(E[m].sum()) if full else np.nan
    m3 = (t > -72) & (t <= 0)
    out["T_past3"] = float(np.nanmean(np.asarray(temp, float)[m3])) if m3.any() else np.nan
    for j in (1, 2):
        m = (t > 24 * (j - 1)) & (t <= 24 * j)
        ok = m.sum() >= 20
        g = lambda a, f: float(f(np.asarray(a, float)[m])) if (ok and a is not None) else np.nan
        out[f"P_{j}"] = float(P[m].sum()) if ok else np.nan
        out[f"ET0_{j}"] = float(E[m].sum()) if ok else np.nan
        out[f"Tmax_{j}"] = g(temp, np.nanmax)
        out[f"RH_{j}"] = g(rh, np.nanmean)
        out[f"SW_{j}"] = g(sw, np.nanmean)
        out[f"U_{j}"] = g(wind, np.nanmean)
    return out


class MLCore:
    """Обгортка над навченими моделями. load() повертає None, якщо моделі/бібліотеки недоступні."""

    def __init__(self, bundle):
        self.models = bundle["models"]
        self.meta = bundle["meta"]

    @classmethod
    def load(cls, path=MODEL_PATH):
        try:
            import warnings

            import joblib
            import sklearn
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")        # попередження про іншу версію sklearn — повідомляємо самі
                core = cls(joblib.load(path))
            note = None
            trained = str(core.meta.get("sklearn_version", ""))
            if trained.split(".")[:2] != sklearn.__version__.split(".")[:2]:
                note = (f"ML-моделі навчено на scikit-learn {trained}, встановлено {sklearn.__version__} — "
                        f"радимо перенавчити: python -m ds.core.train_core")
            return core, note
        except Exception as e:                      # немає sklearn/joblib, файлу або несумісна версія
            return None, f"ML-ядро недоступне ({type(e).__name__}: {str(e)[:80]}) — прогноз лише фізикою"

    @classmethod
    def load_for_evaluation(cls):
        path = os.environ.get("HYDRO_EVALUATION_MODEL")
        if not path:
            raise ValueError("Set HYDRO_EVALUATION_MODEL to an independently trained bundle (training_end_year <= 2023). Production v2 includes 2024–2025 and is not valid for Stage 4 evaluation.")
        core, note = cls.load(path)
        if core is None or core.meta.get("training_end_year", 9999) > 2023:
            raise ValueError("Evaluation bundle must declare training_end_year <= 2023")
        return core, note

    def sigma(self, target, h):
        q = self.meta.get("quality_forecast_2024_2025", {}).get(f"{target}_h{h}", {})
        return float(q.get("sigma_rel", 0.05))

    def predict(self, target, h, feats: dict) -> float:
        import pandas as pd
        cols = feature_names(h)
        x = pd.DataFrame([{c: feats.get(c, np.nan) for c in cols}])
        return float(self.models[(target, h)].predict(x)[0])


def build_features(now, w_now, w_24, w_72, wx_daily: dict, h: int) -> dict:
    f = dict(wx_daily)
    f["w"] = w_now
    f["dw1"] = w_now - w_24 if w_24 is not None and np.isfinite(w_24) else np.nan
    f["dw3"] = w_now - w_72 if w_72 is not None and np.isfinite(w_72) else np.nan
    doy = now.timetuple().tm_yday
    f["doy_sin"], f["doy_cos"] = math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25)
    f["P_h"] = sum(f.get(f"P_{j}", np.nan) for j in range(1, h + 1))
    f["ET0_h"] = sum(f.get(f"ET0_{j}", np.nan) for j in range(1, h + 1))
    return f


def nearest_reading(readings, t, tol_h=3.0, times=None):
    """readings — відсортований список (time, θ); значення найближчого показу в межах ±tol_h, інакше None.
    times — необов'язково готовий список часів (для швидкого пошуку в довгих рядах)."""
    import bisect
    if not readings:
        return None
    ts = times if times is not None else [x[0] for x in readings]
    i = bisect.bisect_left(ts, t)
    best, dmin = None, tol_h * 3600 + 1
    for j in (i - 1, i):
        if 0 <= j < len(readings):
            d = abs((readings[j][0] - t).total_seconds())
            if d < dmin:
                best, dmin = readings[j][1], d
    return best


def profile_weights(depths_m, bottom_m):
    """Вага датчика = товщина шару між серединами сусідніх глибин, обрізана на [0, bottom] (як у Етапі 3)."""
    use = sorted(depths_m)
    if len(use) == 1:
        return {use[0]: 1.0}
    w = {}
    for k, z in enumerate(use):
        top = 0.0 if k == 0 else (use[k - 1] + z) / 2
        bot = bottom_m if k == len(use) - 1 else min((z + use[k + 1]) / 2, bottom_m)
        w[z] = max(bot - top, 0.0)
    s = sum(w.values())
    if s <= 0:
        return {use[0]: 1.0}
    return {z: v / s for z, v in w.items() if v > 0}


def build_profile(series: dict, root_m: float, tol_h: float = 3.0):
    """
    Один узгоджений ряд вологості кореневої зони з кількох датчиків (1.10).
    series — {глибина_м: [(час, θ), …]}. Беруться датчики до глибини коренів + 10 см (якщо таких немає —
    найменш глибокий); для кожного моменту найчисельнішого ряду шукаються покази інших глибин у межах
    ±tol_h і рахується середнє, зважене за товщиною шарів. Моменти, де бракує якоїсь глибини, пропускаються.
    Один датчик — його ряд без змін.
    """
    if not series:
        return []
    depths = sorted(d for d in series if d <= root_m + 0.10) or [min(series)]
    if len(depths) == 1:
        return sorted(series[depths[0]])
    wts = profile_weights(depths, max(root_m, max(depths)))
    depths = [d for d in depths if d in wts]
    times = {d: [t for t, _ in sorted(series[d])] for d in depths}
    ref = max(depths, key=lambda d: len(series[d]))
    out = []
    for t, _ in sorted(series[ref]):
        acc, ok = 0.0, True
        for d in depths:
            v = nearest_reading(sorted(series[d]), t, tol_h, times=times[d])
            if v is None:
                ok = False
                break
            acc += wts[d] * v
        if ok:
            out.append((t, acc))
    return out


def irrigated_between(irr_times, t0, t1):
    """Чи був записаний полив в інтервалі (t0, t1]."""
    return any(t0 < t <= t1 for t in irr_times)


def rollout(core, target, t0, w0, w_m1, w_m3, now, wx):
    """
    Прокрутка відносної вологості від моменту t0 до now кроками по 24 год ML-ядром (горизонт 1 доба)
    за фактичною погодою. wx — dict(times, precip, eto, temp, rh, sw, wind). Повертає (w_now, історія [(t, w)]).
    """
    hist = [(t0 - timedelta(hours=72), w_m3), (t0 - timedelta(hours=24), w_m1), (t0, w0)]
    t, w = t0, w0
    while (now - t).total_seconds() >= 24 * 3600:
        wd = daily_weather(wx["times"], t, wx["precip"], wx["eto"], wx["temp"], wx["rh"], wx["sw"], wx["wind"])
        w24 = next((v for tt, v in reversed(hist) if abs((tt - (t - timedelta(hours=24))).total_seconds()) < 3 * 3600), None)
        w72 = next((v for tt, v in reversed(hist) if abs((tt - (t - timedelta(hours=72))).total_seconds()) < 3 * 3600), None)
        dw = core.predict(target, 1, build_features(t, w, w24, w72, wd, 1))
        t, w = t + timedelta(hours=24), w + dw
        hist.append((t, w))
    if (now - t).total_seconds() > 0:                  # залишок < доби — частка добового кроку
        frac = (now - t).total_seconds() / 86400.0
        wd = daily_weather(wx["times"], t, wx["precip"], wx["eto"], wx["temp"], wx["rh"], wx["sw"], wx["wind"])
        w24 = next((v for tt, v in reversed(hist) if abs((tt - (t - timedelta(hours=24))).total_seconds()) < 3 * 3600), None)
        w = w + frac * core.predict(target, 1, build_features(t, w, w24, None, wd, 1))
        hist.append((now, w))
    return w, hist


def shift_profile(H: int, c24: float, c48: float):
    """Зсув траєкторії дефіциту по годинах: 0 → c24 (24 год) → c48 (48 год), далі сталий."""
    hrs = np.arange(1, H + 1, dtype=float)
    return np.interp(hrs, [0.0, 24.0, 48.0], [0.0, c24, c48], right=c48)


def sd_profile(H: int, s24: float, s48: float):
    hrs = np.arange(1, H + 1, dtype=float)
    return np.interp(hrs, [0.0, 24.0, 48.0], [0.0, s24, s48], right=s48)


# ---------------------------------------------------------------------------- локальна поправка
class LocalResidual:
    """
    Локальне донавчання: невелика ML-модель вчить залишок «факт − прогноз ядра» на даних самого поля.
    Вмикається лише якщо на останніх 25 % прикладів (хронологічно) вона зменшує MAE.
    """
    MIN_ROWS = 60

    @staticmethod
    def fit(rows: list, h: int):
        import pandas as pd
        from sklearn.ensemble import HistGradientBoostingRegressor
        if not rows:
            return None, dict(enabled=False, n=0, reason="немає прикладів із мітками (потрібні покази датчика через 24/48 год)")
        df = pd.DataFrame(rows).sort_values("time")
        df = df[np.isfinite(df["y"]) & np.isfinite(df["core"])]
        if len(df) < LocalResidual.MIN_ROWS:
            return None, dict(enabled=False, n=int(len(df)), reason=f"замало прикладів ({len(df)} < {LocalResidual.MIN_ROWS})")
        cols = feature_names(h) + ["core"]
        k = int(len(df) * 0.75)
        tr, va = df.iloc[:k], df.iloc[k:]
        mk = lambda: HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=300,
                                                   max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                                   random_state=0)
        m = mk().fit(tr[cols], tr["y"] - tr["core"])
        mae_core = float(np.mean(np.abs(va["y"] - va["core"])))
        mae_loc = float(np.mean(np.abs(va["y"] - va["core"] - m.predict(va[cols]))))
        info = dict(n=int(len(df)), mae_core_val=mae_core, mae_local_val=mae_loc, enabled=mae_loc < mae_core)
        if not info["enabled"]:
            info["reason"] = "на відкладених даних поля поправка не зменшила похибку — лишаємо ядро"
            return None, info
        return mk().fit(df[cols], df["y"] - df["core"]), info
