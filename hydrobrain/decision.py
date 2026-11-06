"""
Блок рішення: «поливати зараз / не поливати / відкласти» з урахуванням прогнозу і невизначеності.

Метод: Монте-Карло по горизонту Hz (за замовчуванням 72 год). Для кожного з N сценаріїв
випадково варіюються: стартовий дефіцит (σ з фільтра Калмана), запас вологи TAW (σ_geo),
ETo (σ_meas, росте з часом прогнозу) і опади (члени ансамблю або ймовірність опадів).
Проганяється водний баланс без поливу (S0) → розподіл часу, коли дефіцит перетне поріг RAW.

Правила (прозорі, щоб їх можна було пояснити):
  1. Поза вегетацією → не поливати.
  2. Критичний дефіцит (Ks < ~0.7) → поливати зараз; виняток — сильний дощ у найближчі 3 год.
  3. Стрес уже почався (Dr ≥ RAW) → поливати, якщо дощ з імовірністю ≥ 60% не покриє дефіцит
     протягом 6-12 год (у чутливу фазу — 6 год); для дощування при сильному вітрі/спеці —
     перенести на найближче «добре вікно».
  4. Стресу ще немає → ймовірність стресу протягом горизонту рішення (48 год):
        < 50%  → не поливати (дощ або запас вологи покривають потребу);
                 якщо при цьому ≥ 25% — позначка «прикордонна ситуація» і раніша повторна перевірка;
        ≥ 50%  → запланувати полив на найкраще вікно до медіанного моменту перетину порогу.

Усі пороги зібрано в DECISION_THRESHOLDS нижче (з поясненням, звідки кожен). Їх можна перевизначити
для поля через P["decision_thresholds"]. Пороги не підбиралися на даних — це експертні значення;
перевірка і калібрування — етап бектесту (див. CHANGELOG.md).
  Норма: дефіцит на момент поливу − «запас під дощ» (½ очікуваних ефективних опадів 48 год),
  ÷ ефективність способу, × (1 + норма промивання), × навчений множник користувача.
"""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np

from . import salinity
from .databases import IRRIGATION_METHODS

# Пороги правил рішення. Значення — експертні (не підібрані на даних); у коментарі — логіка вибору.
DECISION_THRESHOLDS = dict(
    h_decision=48,              # год: горизонт, на якому оцінюється ймовірність стресу (≈ 2 доби прогнозу)
    crit_frac=0.3,              # критичний дефіцит = RAW + 0.3·(TAW − RAW); при p≈0.5 це Ks ≈ 0.7
    rescue_critical=0.7,        # частка сценаріїв, у яких дощ за 3 год знімає критичний дефіцит
    rescue_window_h=3,          # год: скільки можна чекати дощ при критичному дефіциті
    stress_window_sensitive_h=6,    # год: скільки чекати дощ, якщо стрес почався у чутливу фазу
    stress_window_h=12,             # те саме поза чутливою фазою
    stress_window_mild_extra_h=18,  # + год, якщо стрес ще легкий (Ks ≥ 0.9)
    rescue_mild=0.5,            # частка сценаріїв «дощ рятує» для легкого стресу (Ks ≥ 0.9)
    rescue_strong=0.65,         # те саме для сильнішого стресу — потрібна більша впевненість
    p_skip=0.5,                 # P(стрес за h_decision) < p_skip → не поливати
    p_watch=0.25,               # p_watch ≤ P < p_skip → «прикордонна ситуація», раніша перевірка
    p_dry_decisive=0.5,         # без дощу P(стрес) ≥ цього → дощ вважається вирішальним
    wind_max=5.0,               # м/с: дощування при сильнішому вітрі переноситься
    heat_tmax=30.0,             # °C: попередження про спеку
    waste_frac=0.4,             # очікувані втрати > 0.4·нетто → норма зменшується
    max_rain_waits=2,           # v1.2: скільки разів поспіль можна «чекати дощ», якщо стрес уже почався
    surface_storage_mm=3.0,     # 1.10: мм поверхневої (западинної) ємності поля між циклами поливу
    max_cycles=8,               # 1.10: максимум циклів «полив — пауза» за одну добу
)


def infiltration_plan(gross_mm, rate_mm_h, ksat_mm_h, storage_mm=3.0, max_cycles=8):
    """
    Полив циклами «полив — пауза» (cycle & soak) для дощування з інтенсивністю вищою за Ksat (1.10).
    Модель, узгоджена з балансом 1.2: під час поливу ґрунт вбирає Ksat мм/год, надлишок (I − Ksat)
    тримається на поверхні до ємності западин S, а в паузі вбирається за S / Ksat год:
      цикл   t_on = S / (I − Ksat), за цикл подається I·t_on мм — і вся вбирається;
      пауза  t_soak = S / Ksat;  циклів n = ⌈брутто / (I·t_on)⌉, не більше max_cycles.
    Безперервно (без циклів) вбиралась би лише частка Ksat / I.
    """
    rate, ks, s = float(rate_mm_h), float(ksat_mm_h), max(float(storage_mm), 0.5)
    gross_mm = float(gross_mm)
    if not all(math.isfinite(v) for v in (rate, ks, s, gross_mm)) or rate <= 0 or ks <= 0 or gross_mm <= 0:
        raise ValueError("План циклів потребує додатних скінченних норми, інтенсивності та Ksat")
    if rate <= ks or int(max_cycles) < 1:
        raise ValueError("Цикли потрібні лише при інтенсивності > Ksat; max_cycles >= 1")
    t_on = s / (rate - ks)
    per = rate * t_on
    soak = s / max(ks, 1e-6)
    need = max(int(math.ceil(float(gross_mm) / max(per, 1e-6) - 1e-9)), 1)
    n = min(need, int(max_cycles))
    deliver = min(float(gross_mm), n * per, 24.0 * ks)
    n = max(1, int(math.ceil(deliver / per - 1e-9)))
    last_mm = max(deliver - (n - 1) * per, 0.0)
    return dict(cycles=n, needed_cycles=need, limited=deliver < gross_mm - 1e-9, on_min=round(min(t_on, deliver / rate) * 60, 3),
                last_on_min=round(last_mm / rate * 60, 3), last_cycle_mm=round(last_mm, 3),
                duration_min=round(deliver / rate * 60, 3),
                soak_min=round(soak * 60, 3), per_cycle_mm=round(per, 1),
                total_h=round(deliver / rate + (n - 1) * soak, 3), deliver_gross_mm=round(deliver, 1), deliver_exact=deliver,
                rate_mm_h=round(rate, 1), ksat_mm_h=round(ks, 2), storage_mm=s,
                continuous_infiltration=round(ks / rate, 3))


_PSTRESS_CAL = {}


def _pstress_map(kind):
    """Крива перекалібрування ймовірності стресу (hydrobrain/models/pstress_calibration.json) або None."""
    if kind not in _PSTRESS_CAL:
        try:
            import json
            import os
            with open(os.path.join(os.path.dirname(__file__), "models", "pstress_calibration.json"),
                      encoding="utf-8") as stream:
                d = json.load(stream)[kind]
            _PSTRESS_CAL[kind] = (np.asarray(d["x"], float), np.asarray(d["y"], float))
        except Exception:
            _PSTRESS_CAL[kind] = None
    return _PSTRESS_CAL[kind]


def _thr(P):
    return {**DECISION_THRESHOLDS, **(P.get("decision_thresholds") or {})}


def sample_rain(precip, prob, day_index, N, rng, members=None):
    T = len(precip)
    if members is not None and members.shape[1] >= T and members.shape[0] >= 5:
        idx = rng.integers(0, members.shape[0], N)
        return np.maximum(members[idx, :T], 0.0)
    precip = np.nan_to_num(np.asarray(precip, float))
    prob = np.asarray(prob, float)
    prob = np.where(np.isfinite(prob), prob, np.where(precip > 0.1, 70.0, 0.0)) / 100.0
    d = day_index - day_index[0]
    nd = int(d.max()) + 1
    U = rng.random((N, nd))[:, d]
    mult = rng.lognormal(-0.125, 0.5, (N, nd))[:, d]
    amount = np.where(precip > 0.05, precip, np.where(prob >= 0.3, 0.4, 0.0))
    return np.where(U < prob[None, :], amount[None, :] * mult, 0.0)


def _first_cross(dr, thr):
    hit = dr >= thr
    any_hit = hit.any(axis=1)
    idx = np.argmax(hit, axis=1).astype(float)
    idx[~any_hit] = 1e6          # «не перетне в межах горизонту»
    return idx + 1.0  # годин від «зараз»


def best_window(S, start, end, method, prob, preferred_hours, avoid_rain=True):
    """Найкраща година для поливу: мала ETo, слабкий вітер (дощування), без дощу, дозволена година."""
    end = min(end, len(S.times))
    if end <= start:
        return start
    meth = IRRIGATION_METHODS[method]
    pr = np.nan_to_num(np.asarray(prob, float))
    best, best_score = start, 1e9
    for i in range(start, end):
        score = S.eto[i] * 10.0
        if meth["wind_sensitive"]:
            score += max(0.0, S.u2[i] - 3.0) * 2.0 + S.drift[i] * 20.0
        if avoid_rain:
            score += pr[i:i + 3].max(initial=0.0) / 25.0
        if preferred_hours and S.local_hour[i] not in preferred_hours:
            score += 3.0
        score += 0.01 * (i - start)  # за інших рівних — раніше
        if score < best_score:
            best, best_score = i, score
    return best


def decide(engine, S, state, sigma_dr, precip, prob, P, user_mult=1.0, k_learn=1.0,
           members=None, rng=None, sensor_ok=True, sensor_fresh_h=None, dr_shift=None, dr_shift_sd=None):
    """
    dr_shift / dr_shift_sd — необов'язковий зсув траєкторій дефіциту (мм, по годинах горизонту) від ML-ядра
    і його невизначеність: природна (без поливу) траєкторія береться з ML, а фізика дає ефект поливу і
    розкид сценаріїв (див. ml_forecast.py). Без них — як раніше, лише фізика.
    """
    rng = rng or np.random.default_rng(42)
    TH = _thr(P)
    N = int(P.get("mc_samples", 200))
    H = len(S.times)
    crop = engine.crop
    meth = engine.method
    reasons = []

    # ---------- сценарії
    sat_cap = engine.sat_unit * S.zr[0]
    taw_mult = np.clip(rng.lognormal(0.0, float(P.get("sigma_geo", 0.2)), N), 0.5, 1.7)
    taw0 = engine.taw(S.zr[0], taw_mult)
    dr0 = np.clip(rng.normal(state["dr"], max(sigma_dr, 0.5), N), -sat_cap, taw0)
    lead_days = np.arange(H) / 24.0
    et_mult = np.clip(1.0 + rng.normal(0, 1, (N, 1)) * (float(P.get("sigma_meas", 0.08)) + 0.01 * lead_days[None, :]),
                      0.5, 1.6)
    rain = sample_rain(precip, prob, S.day_index, N, rng, members)
    zeros = np.zeros((N, H))

    def new_state():
        du0 = state.get("du")
        if du0 is not None:   # верхній шар — у тій самій пропорції до загального дефіциту
            dr_m = float(state["dr"])
            du0 = np.clip(dr0 * (du0 / dr_m if dr_m > 1.0 else 1.0), 0, None) if dr_m > 1.0 else np.zeros(N)
        st = engine.init_state(N, dr0, x0=state.get("x", 0.0), zr0=state.get("zr"), du0=du0,
                               swe0=float(state.get("swe", 0.0)))
        st["de"] = np.full(N, float(state.get("de", engine.rew)))
        st["fw"] = np.full(N, float(state.get("fw", 1.0)))
        return st

    _, r0 = engine.run(S, new_state(), rain, zeros, et_mult, taw_mult, learn_k=k_learn)
    shift = None
    if dr_shift is not None:
        z = rng.standard_normal(N)
        sd = np.zeros(H) if dr_shift_sd is None else np.asarray(dr_shift_sd, float)[:H]
        shift = np.asarray(dr_shift, float)[None, :H] + z[:, None] * sd[None, :]

        def _apply(r):
            r["dr"] = np.minimum(np.maximum(r["dr"] + shift, -sat_cap), r["taw"])
        _apply(r0)
    t_cross = _first_cross(r0["dr"], r0["raw"])
    t_crit = _first_cross(r0["dr"], r0["raw"] + TH["crit_frac"] * (r0["taw"] - r0["raw"]))
    hz = min(H, int(P.get("horizon_h", 72)))
    h_dec = min(int(TH["h_decision"]), hz)
    p_stress = {h: float(np.mean(t_cross <= h)) for h in {12, 24, 48, hz, h_dec}}
    # 1.11: перекалібрування ймовірності (ізотонічна крива з бектесту на ERA5-Land, ds/stage4) — сирі частки
    # сценаріїв Монте-Карло систематично не збігались із частотою подій; пороги рішення діють на калібровані
    p_stress_raw = dict(p_stress)
    # The fitted curve targets 48 h only; it is not a calibration for 12/24/72 h.
    from .ml_forecast import in_domain
    calibration_scope = (H >= 48 and P.get("crop") == "maize" and P.get("irrigation_method") == "drip"
                         and in_domain(P["lat"], P["lon"])[0])
    cal_map = _pstress_map("ml" if dr_shift is not None else "physics") if (
        P.get("pstress_calibration", True) and calibration_scope) else None
    if cal_map is not None:
        p_stress[48] = float(np.interp(p_stress[48], cal_map[0], cal_map[1]))
    p_crit_24 = float(np.mean(t_crit <= 24))

    dr_now = float(state["dr"])
    taw_now = float(engine.taw(S.zr[0]))
    raw_now = float(S.p[0] * taw_now)
    crit_now = raw_now + TH["crit_frac"] * (taw_now - raw_now)
    ks_now = 1.0 if dr_now <= raw_now else max(0.0, (taw_now - dr_now) / max((1 - S.p[0]) * taw_now, 1e-6))
    rain_sum = lambda a, b: rain[:, a:min(b, H)].sum(axis=1)
    pe_sum = lambda a, b: r0["pe"][:, a:min(b, H)].sum(axis=1)
    rain_stats = dict(exp_24h=float(rain_sum(0, 24).mean()), exp_48h=float(rain_sum(0, 48).mean()),
                      exp_72h=float(rain_sum(0, 72).mean()), p_ge5_48h=float(np.mean(rain_sum(0, 48) >= 5.0)),
                      max_prob_24h=float(np.nanmax(np.r_[np.asarray(prob[:24], float), 0.0])))
    wet_hours = np.nonzero(np.nan_to_num(np.asarray(precip[:hz], float)) >= 0.3)[0]
    rain_stats["first_rain_h"] = int(wet_hours[0]) + 1 if len(wet_hours) else None
    dvs_now = float(S.dvs[0])
    sens = crop.get("sensitive", (0, 0))
    in_sensitive = sens[1] > sens[0] and sens[0] <= dvs_now <= sens[1]
    preferred = P.get("preferred_hours") or []

    action, t_idx, next_check = "skip", None, 24
    postpone_h = None   # відкладення без запланованої норми (чекаємо дощ, потім перерахунок)
    perennial = P["crop"] in ("lawn", "grape", "apple")

    if (dvs_now <= 0.0 or dvs_now > 1.02) and not perennial:
        reasons.append(dict(code="off_season", dvs=dvs_now))
        action, next_check = "skip", 24
    elif dr_now >= crit_now:
        reasons.append(dict(code="critical_deficit", dr=dr_now, raw=raw_now, taw=taw_now, ks=ks_now))
        # дощ «рятує», якщо протягом 3 год повертає дефіцит нижче порогу RAW
        rw = int(TH["rescue_window_h"])
        need = dr_now - r0["raw"][:, 0] + r0["et"][:, :rw].sum(axis=1)
        rescue = float(np.mean(pe_sum(0, rw) >= np.maximum(need, 1.0)))
        waits = int(P.get("_rain_waits", 0))
        if rescue >= TH["rescue_critical"] and waits >= TH["max_rain_waits"]:
            reasons.append(dict(code="rain_wait_limit", n=waits))
            rescue = 0.0
        if rescue >= TH["rescue_critical"]:
            reasons.append(dict(code="rain_rescue", p=rescue, hours=rw, mm=float(rain_sum(0, rw).mean())))
            action, t_idx, next_check, postpone_h = "postpone", None, rw, rw
        else:
            action, t_idx = "irrigate_now", 0
    elif dr_now >= raw_now:
        reasons.append(dict(code="stress_started", dr=dr_now, raw=raw_now, ks=ks_now))
        # легкий стрес (Ks ≥ 0.9) можна «перетерпіти» довше, у чутливу фазу — коротше
        win = ((TH["stress_window_sensitive_h"] if in_sensitive else TH["stress_window_h"])
               + (TH["stress_window_mild_extra_h"] if ks_now >= 0.9 else 0))
        win = int(win)
        win = min(win, H)
        # дощ «рятує», якщо ефективні опади за вікно покривають поточний дефіцит понад 0.9·RAW
        # плюс ETc за цей час (для кожного сценарію окремо; без дощу — ніколи)
        need = dr_now - 0.9 * r0["raw"][:, 0] + r0["et"][:, :win].sum(axis=1)
        rescue = float(np.mean(pe_sum(0, win) >= np.maximum(need, 1.0)))
        waits = int(P.get("_rain_waits", 0))
        if rescue >= (TH["rescue_mild"] if ks_now >= 0.9 else TH["rescue_strong"]) and waits >= TH["max_rain_waits"]:
            reasons.append(dict(code="rain_wait_limit", n=waits))
            rescue = 0.0
        if rescue >= (TH["rescue_mild"] if ks_now >= 0.9 else TH["rescue_strong"]):
            first = rain_stats["first_rain_h"] if rain_stats["first_rain_h"] is not None else win
            reasons.append(dict(code="rain_rescue", p=rescue, hours=first, mm=float(rain_sum(0, win).mean())))
            wait = int(min(win, first + 3))
            action, t_idx, next_check, postpone_h = "postpone", None, max(wait, 3), max(wait, 3)
            reasons.append(dict(code="recheck_after_rain", hours=wait))
        else:
            if rain_stats["exp_24h"] >= 2.0:
                reasons.append(dict(code="rain_insufficient", mm=rain_stats["exp_24h"], hours=rain_stats["first_rain_h"]))
            j = best_window(S, 0, min(5, H), P["irrigation_method"], prob, preferred)
            if j <= 1:
                action, t_idx = "irrigate_now", 0
            else:
                action, t_idx = "postpone", j
                reasons.append(dict(code="better_window", hours=j + 1))
    else:
        hrs = t_cross[np.isfinite(t_cross)]
        t_med = float(np.median(t_cross))
        if p_stress[h_dec] < TH["p_skip"]:
            # чи саме дощ «рятує»? — порівнюємо зі сценарієм без опадів
            _, rd = engine.run(S, new_state(), zeros, zeros, et_mult, taw_mult, learn_k=k_learn)
            p_dry = float(np.mean(_first_cross(rd["dr"], rd["raw"]) <= h_dec))
            if p_dry >= TH["p_dry_decisive"] and rain_stats["exp_48h"] >= 1.0:
                reasons.append(dict(code="rain_decisive", mm=rain_stats["exp_48h"], p_dry=p_dry,
                                    p=rain_stats["max_prob_24h"] / 100.0, hours=rain_stats["first_rain_h"]))
            elif rain_stats["exp_48h"] >= 3.0:
                reasons.append(dict(code="rain_expected", mm=rain_stats["exp_48h"],
                                    p=rain_stats["max_prob_24h"] / 100.0, hours=rain_stats["first_rain_h"]))
            reasons.append(dict(code="enough_water", hours=t_med if t_med <= H else None, dr=dr_now, raw=raw_now, taw=taw_now))
            q25 = float(np.quantile(t_cross, 0.25))
            action, next_check = "skip", int(np.clip(q25 * 0.7, 3, 24))
            if p_stress[h_dec] >= TH["p_watch"]:
                reasons.append(dict(code="watch", p=p_stress[h_dec], hours=next_check, horizon_h=h_dec))
        else:
            # плануємо полив на найкраще вікно до медіанного часу перетину порогу
            q_deadline = float(np.interp(TH["p_skip"], cal_map[1], cal_map[0])) if cal_map is not None else 0.5
            t50 = float(np.quantile(t_cross, min(0.5, q_deadline)))
            deadline = int(np.clip(t50, 1, h_dec))
            reasons.append(dict(code="will_need_soon", hours=deadline, p=p_stress[h_dec], dr=dr_now, raw=raw_now, horizon_h=h_dec))
            j = best_window(S, 0, deadline, P["irrigation_method"], prob, preferred)
            if j <= 1:
                action, t_idx = "irrigate_now", 0
            else:
                action, t_idx = "postpone", j
                reasons.append(dict(code="better_window", hours=j + 1))
        if rain_stats["exp_24h"] >= 2.0 and action != "skip":
            reasons.append(dict(code="rain_expected", mm=rain_stats["exp_24h"], p=rain_stats["max_prob_24h"] / 100.0,
                                hours=rain_stats["first_rain_h"]))

    # ---------- погодні застереження в момент поливу
    if t_idx is not None and action in ("irrigate_now", "postpone"):
        if meth["wind_sensitive"] and S.u2[t_idx] > TH["wind_max"] and dr_now < crit_now:
            j = best_window(S, t_idx, min(t_idx + 12, H), P["irrigation_method"], prob, preferred)
            if j != t_idx:
                reasons.append(dict(code="wind_high", wind=float(S.u2[t_idx]), hours=j + 1))
                action, t_idx = "postpone", j
        tmax24 = float(np.max(S.t_air[:24]))
        if tmax24 >= TH["heat_tmax"]:
            reasons.append(dict(code="heat", tmax=tmax24))
    if in_sensitive and action != "skip":
        reasons.append(dict(code="sensitive_stage", dvs=dvs_now))

    # ---------- норма поливу
    amount = None
    expected_waste = 0.0
    if action in ("irrigate_now", "postpone") and t_idx is not None:
        t_idx = int(min(t_idx, H - 1))
        dr_at = float(np.median(r0["dr"][:, t_idx])) if t_idx > 0 else dr_now
        taw_at = float(engine.taw(S.zr[t_idx]))
        buffer = min(0.5 * float(np.mean(pe_sum(t_idx, t_idx + 48))), 0.3 * taw_at)
        net = max(dr_at - buffer, 0.0)
        net = min(net, 0.95 * taw_at)
        cap = float(P.get("max_single_mm") or meth.get("max_mm", 40.0))
        if net > cap:
            reasons.append(dict(code="split", total=net, now=cap))
            net = cap
        net = max(net, meth["min_mm"]) if dr_at > meth["min_mm"] else max(net, dr_at)
        ece_now = float(engine.ece_base + state.get("x", 0.0))
        lr = 0.0
        if P.get("allow_leaching", True) and ece_now > 0.8 * crop["ece_thr"]:
            lr = salinity.leaching_requirement(float(P.get("ec_iw", 0.3)), crop["ece_thr"], P["irrigation_method"])
            reasons.append(dict(code="salinity", ece=ece_now, thr=crop["ece_thr"], lr=lr))
        eff = meth["ea"] * float(P.get("ei_mult", 1.0)) * (1.0 - float(S.drift[t_idx]))
        if S.drift[t_idx] > 0.08:
            reasons.append(dict(code="wind_losses", loss=float(S.drift[t_idx]), wind=float(S.u2[t_idx])))
        gross = net / max(eff, 0.3) * (1.0 + lr) * user_mult
        # ---- інфільтрація (1.10): якщо інтенсивність дощування перевищує Ksat, безперервний полив стікає з поля.
        # Норма розраховується на полив циклами «полив — пауза», за яких уся вода вбирається; якщо циклів
        # забагато, за добу дається стільки, скільки вбереться, решта — наступного разу.
        plan = None
        if engine.infiltration_fraction() < 1.0:
            plan = infiltration_plan(gross, engine.application_rate(), engine.hyd["ksat_mm_h"],
                                     float(P.get("surface_storage_mm") or TH["surface_storage_mm"]),
                                     int(TH["max_cycles"]))
            if plan["limited"]:
                k_inf = plan["deliver_exact"] / max(gross, 1e-6)
                reasons.append(dict(code="infiltration_limited", now=plan["deliver_gross_mm"], total=gross,
                                    rate=plan["rate_mm_h"], ksat=plan["ksat_mm_h"], cycles=plan["cycles"]))
                gross *= k_inf
                net *= k_inf
            else:
                reasons.append(dict(code="infiltration_cycles", **{k: plan[k] for k in
                                    ("cycles", "on_min", "soak_min", "total_h", "rate_mm_h", "ksat_mm_h")}))
        # перевірка втрат: прогін зі зробленим поливом (циклами — уся вода вбирається; узгоджено з планом)
        irr = np.zeros((N, H))
        if plan is None:
            irr[:, t_idx] = gross
        else:
            # Cycle/soak keeps the surface reservoir wet; infiltration proceeds at Ksat
            # until the last stored water drains, including the final partial cycle.
            ks = engine.hyd["ksat_mm_h"]
            elapsed = np.arange(H - t_idx, dtype=float)
            irr[:, t_idx:] = np.clip(gross - elapsed * ks, 0.0, ks)
        irr_infil = None if plan is None else np.ones((N, H))
        _, r1 = engine.run(S, new_state(), rain, irr, et_mult, taw_mult, learn_k=k_learn, irr_infil=irr_infil)
        if shift is not None:
            _apply(r1)
        waste = (r1["dp"].sum(1) + r1["ro"].sum(1)) - (r0["dp"].sum(1) + r0["ro"].sum(1))
        expected_waste = float(np.mean(waste))
        if expected_waste > TH["waste_frac"] * net and net > meth["min_mm"]:
            k = max(0.5, 1.0 - (expected_waste - 0.2 * net) / max(net, 1e-6))
            gross *= k
            net *= k
            rain_48 = float(np.mean(pe_sum(t_idx, t_idx + 48)))
            # 1.10: причину називаємо правильно — дощ лише тоді, коли він справді очікується
            reasons.append(dict(code="reduced_for_rain" if rain_48 >= 1.0 else "reduced_for_drainage",
                                factor=k, waste=expected_waste, rain_mm=rain_48))
        if abs(user_mult - 1.0) > 0.03:
            reasons.append(dict(code="learned_adjust", mult=user_mult))
        cap_up = P.get("user_mult_max")
        if cap_up and user_mult >= cap_up - 0.01:
            # користувач систематично ллє більше, ніж модель вважає потрібним, а множник уже на межі
            reasons.append(dict(code="user_overwater", mult=user_mult))
        area = P.get("area_m2")
        amount = dict(net_mm=round(net, 1), gross_mm=round(gross, 1), liters_per_m2=round(gross, 1),
                      leaching_fraction=round(lr, 3), user_multiplier=round(user_mult, 3),
                      efficiency=round(eff, 3))
        if plan is not None:     # план циклів для остаточної норми (після всіх поправок)
            p2 = infiltration_plan(gross, engine.application_rate(), engine.hyd["ksat_mm_h"],
                                   plan["storage_mm"], 10 ** 6)
            for reason in reasons:
                if reason["code"] == "infiltration_cycles":
                    reason.update({k: p2[k] for k in ("cycles", "on_min", "soak_min", "total_h", "last_on_min")})
            amount["cycles"] = {k: p2[k] for k in ("cycles", "on_min", "soak_min", "per_cycle_mm", "total_h",
                                                     "rate_mm_h", "ksat_mm_h", "continuous_infiltration",
                                                     "last_on_min", "last_cycle_mm", "duration_min")}
        # розкид норми за сценаріями (10–90%): та сама логіка, що й для медіани, але для кожного сценарію
        # (використовується для орієнтовної рекомендації без датчика, 1.8)
        q = np.quantile(r0["dr"][:, t_idx], [0.1, 0.9])
        nq = np.clip(q - buffer, 0.0, min(0.95 * taw_at, cap))
        ratio = gross / max(net, 1e-6)
        amount["gross_p10_p90"] = [round(float(nq[0] * ratio), 1), round(float(nq[1] * ratio), 1)]
        if area:
            amount["m3_total"] = round(gross * float(area) / 1000.0, 2)
            if P.get("flow_rate_m3_h"):
                amount["duration_min"] = int(round((gross * float(area) / 1000.0) / float(P["flow_rate_m3_h"]) * 60))
        if plan is not None:     # чистий час поливу в циклах; загальний час разом із паузами — cycles.total_h
            amount["duration_min"] = round(p2["duration_min"], 3)
        next_check = max(int(t_idx) + 1, 1) if action == "postpone" else 24

    # ---------- впевненість
    p = p_stress[h_dec]
    margin = abs(p - 0.5) * 2 if action != "irrigate_now" else min(1.0, max(0.0, (dr_now - raw_now) / max(raw_now, 1) + 0.5))
    conf = 0.55 * margin + 0.25 * (1.0 if sensor_ok and sensor_fresh_h is not None and sensor_fresh_h < 48 else 0.4) \
        + 0.20 * (1.0 - min(1.0, sigma_dr / max(raw_now, 1.0)))
    conf_label = "висока" if conf >= 0.7 else ("середня" if conf >= 0.45 else "низька")

    t_local = S.times[t_idx] if t_idx is not None else None
    return dict(action=action, t_idx=t_idx, when_utc=t_local.isoformat() if t_local else None,
                postpone_hours=(int(t_idx) + 1 if t_idx is not None else int(postpone_h or 0)) if action == "postpone" else 0,
                amount=amount, waiting_for_rain=postpone_h is not None,
                next_check_hours=int(next_check),
                risk=dict(p_stress_12h=round(p_stress[12], 3) if H >= 12 else None,
                          p_stress_24h=round(p_stress[24], 3) if H >= 24 else None,
                          p_stress_48h=round(p_stress[48], 3) if H >= 48 else None,
                          p_stress_horizon=round(p_stress[hz], 3), p_critical_24h=round(p_crit_24, 3) if H >= 24 else None,
                          median_hours_to_threshold=None if np.median(t_cross) > H else float(np.median(t_cross)),
                          expected_waste_mm=round(expected_waste, 1),
                          p_stress_48h_raw=round(p_stress_raw[48], 3) if H >= 48 else None,
                          calibrated=cal_map is not None, calibrated_horizons=[48] if cal_map is not None else [],
                          calibration_scope="maize/drip/Ukraine/48h",
                          probability_horizons_available=[h for h in (12, 24, 48) if h <= H],
                          p_stress_12h_raw=round(p_stress_raw[12], 3),
                          p_stress_24h_raw=round(p_stress_raw[24], 3),
                          p_stress_horizon_raw=round(p_stress_raw[hz], 3)),
                rain=rain_stats, confidence=round(float(conf), 2), confidence_label=conf_label,
                reasons=reasons, state=dict(dr=dr_now, raw=raw_now, taw=taw_now, ks=ks_now, dvs=dvs_now,
                                            sensitive=in_sensitive),
                trajectory=dict(dr_median=np.median(r0["dr"], axis=0).round(2).tolist(),
                                dr_p10=np.quantile(r0["dr"], 0.1, axis=0).round(2).tolist(),
                                dr_p90=np.quantile(r0["dr"], 0.9, axis=0).round(2).tolist(),
                                raw_median=np.median(r0["raw"], axis=0).round(2).tolist()))
