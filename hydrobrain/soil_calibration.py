"""
Калібрування водоутримувальних констант ґрунту (θFC, θWP) за показами датчика вологості.

Навіщо. Довідникові константи (Saxton–Rawls, HWSD, таблиці ґрунтів) можуть сильно відрізнятися від
реального ґрунту в точці датчика. На датчиках ISMN (Етап 3) довідник дав для піску θFC ≈ 0.13, а
датчики показували до 0.35: модель вважала резервуар у кілька разів меншим і реагувала в ~5 разів
сильніше за реальний ґрунт. Фільтр Калмана калібрує поправку Kc і зміщення датчика, але не розмір
резервуара — тому потрібне це окреме калібрування.

Метод (прозорі правила, без підгонки під метрику):
  θFC — за визначенням польової вологоємності: вологість, до якої ґрунт стікає приблизно за 2 доби
        після сильного зволоження. Шукаємо «події зволоження» (різке зростання θ), беремо θ через
        48 год після піку (якщо за цей час не було нового дощу) і медіану по найсильніших подіях
        (пік у верхній половині), бо слабкий дощ на сухий ґрунт не доводить його до FC.
  θWP — нижня межа сухих періодів (1-й перцентиль), але ЛИШЕ якщо ґрунт висихав сильніше за
        довідникове значення; якщо ні, даних про в'янення немає і лишається довідник.
  Довіра до даних росте з кількістю подій: θFC = (n·θFC_дані + k·θFC_довідник) / (n + k), k = 2.
Обмеження: правила розраховані на датчик у кореневій зоні; дуже поверхневий датчик (≤ 5 см)
пересихає сильніше за корені, тож оцінка θWP з нього буде заниженою.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR_WEIGHT = 2.0       # «вага» довідника в подіях зволоження
FC_LAG_H = 48            # год після піку: стандартне визначення FC (стікання 1–3 доби)
MIN_DAYS_WP = 30         # мінімум діб спостережень, щоб говорити про нижню межу


def _hourly(times, values):
    s = pd.Series(np.asarray(values, float), index=pd.DatetimeIndex(pd.to_datetime(times, utc=True)))
    s = s[~s.index.duplicated()].sort_index().dropna()
    if s.empty:
        return s
    return s.resample("1h").mean().interpolate(limit=3, limit_area="inside")


def estimate_fc_wp(times, theta, prior_fc: float, prior_wp: float, rain_times=None, rain_mm=None,
                   rise_thr: float | None = None) -> dict:
    """
    times/theta — покази датчика (будь-яка частота; θ у м³/м³ або %).
    rain_times/rain_mm — погодинні опади+полив (необов'язково; без них події шукаються лише за зростанням θ).
    Повертає dict: theta_fc, theta_wp, n_events, fc_samples, fc_from_data, wp_from_data, notes.
    """
    th = np.asarray(theta, float)
    if np.nanmax(th) > 1.0:
        th = th / 100.0
    s = _hourly(times, th)
    notes = []
    out = dict(theta_fc=float(prior_fc), theta_wp=float(prior_wp), n_events=0, fc_samples=[],
               fc_from_data=None, wp_from_data=None, n_days=0, notes=notes)
    if len(s) < 48:
        notes.append("замало показів датчика (< 2 діб) — лишаємо довідникові значення")
        return out
    out["n_days"] = int(s.notna().sum() // 24)
    rain = None
    if rain_times is not None and rain_mm is not None:
        rain = _hourly(rain_times, rain_mm).reindex(s.index).fillna(0.0)
    thr = rise_thr if rise_thr is not None else max(0.02, 0.1 * (prior_fc - prior_wp))
    # --- події зволоження: θ зросла на ≥ thr відносно мінімуму попередніх 12 год
    lo12 = s.rolling(12, min_periods=3).min()
    rise = (s - lo12) >= thr
    cand = s.index[rise.fillna(False).to_numpy()]
    events, last = [], None
    for t in cand:
        if last is None or (t - last) > pd.Timedelta(hours=24):
            events.append(t)
        last = t
    samples, peaks = [], []
    for t0 in events:
        win = s.loc[t0: t0 + pd.Timedelta(hours=24)]
        if win.dropna().empty:
            continue
        tp = win.idxmax()
        t_fc = tp + pd.Timedelta(hours=FC_LAG_H)
        near = s.loc[t_fc - pd.Timedelta(hours=2): t_fc + pd.Timedelta(hours=2)].dropna()
        if near.empty:
            continue
        if rain is not None and rain.loc[tp + pd.Timedelta(hours=1): t_fc].sum() >= 1.0:
            continue                                   # новий дощ — ґрунт не встиг стекти
        between = s.loc[tp: t_fc].dropna()
        if len(between) and between.iloc[-1] > s.loc[tp] + 0.005:
            continue                                   # вологість зростала далі — не схоже на стікання
        samples.append(float(near.mean()))
        peaks.append(float(s.loc[tp]))
    if samples:
        peaks, samples = np.array(peaks), np.array(samples)
        strong = samples[peaks >= np.median(peaks)]    # найсильніші зволоження
        fc_obs = float(np.median(strong))
        n = len(strong)
        fc = (n * fc_obs + PRIOR_WEIGHT * prior_fc) / (n + PRIOR_WEIGHT)
        out.update(theta_fc=float(fc), fc_from_data=fc_obs, n_events=int(n),
                   fc_samples=[round(float(x), 4) for x in strong])
        notes.append(f"θFC: {n} сильних подій зволоження, θ через {FC_LAG_H} год після піку ≈ {fc_obs:.3f}")
    else:
        notes.append("θFC: не знайдено подій зволоження зі стіканням — лишаємо довідник")
    if out["n_days"] >= MIN_DAYS_WP:
        low = float(np.nanpercentile(s, 1))
        out["wp_from_data"] = low
        if low < prior_wp:
            out["theta_wp"] = low
            notes.append(f"θWP: ґрунт висихав до {low:.3f} < довідника {prior_wp:.3f} — беремо {low:.3f}")
        else:
            notes.append(f"θWP: ґрунт не висихав нижче {low:.3f} (довідник {prior_wp:.3f}) — даних про в'янення немає")
    else:
        notes.append(f"θWP: замало діб ({out['n_days']} < {MIN_DAYS_WP}) — лишаємо довідник")
    if out["theta_fc"] - out["theta_wp"] < 0.03:          # фізична узгодженість
        out["theta_wp"] = out["theta_fc"] - 0.03
        notes.append("θWP обмежено: θFC − θWP ≥ 0.03")
    return out
