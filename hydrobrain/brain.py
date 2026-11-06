"""
HydroBrain — «мозок» системи поливу. Єдина точка входу для інтерфейсу.

    from hydrobrain import HydroBrain
    brain = HydroBrain("hydro_data")
    form  = brain.input_schema("beginner")                 # поля для форми
    field = brain.create_field({...}, mode="beginner")     # створити поле
    brain.add_sensor_reading(field["field_id"], theta=0.27, depth_cm=20)
    rec   = brain.recommend(field["field_id"])             # рекомендація (JSON-сумісний dict)
    brain.log_irrigation(field["field_id"], amount_mm=25)  # факт поливу (і навчання на правках)

Стан кожного поля (дефіцит вологи, фільтр Калмана, події, навчені параметри) зберігається у
<data_dir>/fields/<field_id>.json, тому мозок можна викликати хоч раз на годину, хоч раз на тиждень —
він «доганяє» історію за фактичною погодою.
"""
from __future__ import annotations

import json
import math
import os
import re
from functools import wraps
import threading
import uuid
from datetime import date, datetime, timedelta, timezone

import numpy as np

from . import crop as cropm
from . import explain, params as paramsm
from .assimilation import SoilMoistureKalman
from .balance import WaterBalanceEngine
from .databases import CROPS, IRRIGATION_METHODS
from .decision import decide
from .learning import FeatureEngineer, UserPreferenceLearner, append_training_row
from . import ml_forecast as mlf
from . import meteo
from .soil_calibration import estimate_fc_wp
from .sources import DataSources, SourceError

try:
    from zoneinfo import ZoneInfo
    KYIV = ZoneInfo("Europe/Kyiv")
except Exception:  # pragma: no cover
    KYIV = None


_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def _locked(method):
    @wraps(method)
    def wrapped(self, fid, *args, **kwargs):
        with self.lock(fid):
            return method(self, fid, *args, **kwargs)
    return wrapped


def _number(value, name, minimum=0.0, maximum=None, positive=False):
    try:
        if isinstance(value, bool):
            raise ValueError
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name}: потрібне число") from None
    if not math.isfinite(v) or v < minimum or (positive and v == 0) or (maximum is not None and v > maximum):
        raise ValueError(f"{name}: недопустиме значення")
    return v


def _utc(dt) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KYIV or timezone(timedelta(hours=3)))
    return dt.astimezone(timezone.utc)


def _tz_offset(dt_utc: datetime) -> float:
    if KYIV is None:
        return 3.0
    return dt_utc.astimezone(KYIV).utcoffset().total_seconds() / 3600.0


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    return o


class HydroBrain:
    def __init__(self, data_dir="hydro_data", offline=False, sources: DataSources | None = None, seed=42,
                 no_sensor_policy: str | None = None):
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, "fields"), exist_ok=True)
        self.sources = sources or DataSources(os.path.join(data_dir, "cache"), offline=offline)
        self.fe = FeatureEngineer()
        self.seed = seed
        self._ml_core, self._ml_core_note = (None, None)
        self._ml_loaded = False
        # що робити, коли на полі немає датчика (CHANGELOG 1.8): "orientative" (за замовчуванням) — орієнтовна
        # рекомендація (діапазон норми, низька впевненість, порада виміряти); "off" — рекомендації без датчика
        # вимкнено, лише прохання зробити вимір; "full" — як до 1.8 (повна рекомендація з точною нормою)
        self.no_sensor_policy = os.environ.get("HYDRO_NO_SENSOR_POLICY", "orientative") \
            if no_sensor_policy is None else no_sensor_policy
        if self.no_sensor_policy not in ("orientative", "off", "full"):
            raise ValueError(f"no_sensor_policy має бути orientative / off / full, а не {self.no_sensor_policy!r}")

    @property
    def ml_core(self):
        if not self._ml_loaded:
            self._ml_core, self._ml_core_note = mlf.MLCore.load()
            self._ml_loaded = True
        return self._ml_core

    # ------------------------------------------------------------------ збереження
    def _path(self, fid):
        if not isinstance(fid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", fid):
            raise ValueError("Некоректний ідентифікатор поля")
        return os.path.join(self.data_dir, "fields", f"{fid}.json")

    def load_field(self, fid) -> dict:
        with open(self._path(fid), encoding="utf-8") as f:
            return json.load(f)

    def _save(self, f):
        """Атомарний запис (v1.2): спершу тимчасовий файл, потім os.replace — файл стану ніколи не буває «наполовину»."""
        path = self._path(f["field_id"])
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(f), fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    def lock(self, fid):
        """Блокування поля в межах процесу (для HTTP-сервера з кількома потоками)."""
        key = os.path.abspath(self._path(fid))
        with _LOCKS_GUARD:
            return _LOCKS.setdefault(key, threading.RLock())

    def list_fields(self):
        return [p[:-5] for p in os.listdir(os.path.join(self.data_dir, "fields")) if p.endswith(".json")]

    # ------------------------------------------------------------------ форми
    @staticmethod
    def input_schema(mode="beginner"):
        return paramsm.input_schema(mode)

    @staticmethod
    def catalog():
        """Довідники для випадних списків інтерфейсу."""
        from .databases import FERTILIZERS, GEOLOGY, SOILS_UA
        return dict(crops={k: v["name_uk"] for k, v in CROPS.items()},
                    soils={k: v["name_uk"] for k, v in SOILS_UA.items()},
                    irrigation_methods={k: v["name_uk"] for k, v in IRRIGATION_METHODS.items()},
                    fertilizers={k: v["name_uk"] for k, v in FERTILIZERS.items()},
                    geology={k: v["name_uk"] for k, v in GEOLOGY.items()})

    # ------------------------------------------------------------------ поля
    def create_field(self, inputs: dict, mode="beginner", field_id=None, name=None):
        res = paramsm.Resolver(self.sources)
        P, prov, warnings = res.resolve(inputs, mode)
        eng = WaterBalanceEngine(P)
        fid = field_id or uuid.uuid4().hex[:10]
        self._path(fid)
        if os.path.exists(self._path(fid)):
            raise ValueError("Поле з таким ідентифікатором уже існує")
        f = dict(field_id=fid, name=name or inputs.get("name") or fid, mode=mode, inputs=inputs,
                 params=P, provenance=prov, warnings=warnings, soil_hydraulics=eng.summary(),
                 created=datetime.now(timezone.utc).isoformat(), state=None,
                 events=dict(irrigation=[], sensor=[], fertilizer=[]),
                 learners=dict(user=None), last_recommendation=None)
        for fe in P.get("fertilizers") or []:
            f["events"]["fertilizer"].append(fe)
        self._save(f)
        return dict(field_id=fid, params=P, provenance=prov, warnings=warnings,
                    soil_hydraulics=f["soil_hydraulics"])

    @_locked
    def update_params(self, fid, overrides: dict, mode=None):
        f = self.load_field(fid)
        mode = mode or f["mode"]
        paramsm.validate_mode(mode)
        if not isinstance(overrides, dict):
            raise ValueError("Зміни параметрів мають бути об’єктом")
        # Keep existing expert inputs when a beginner edits only visible fields.
        allowed = {p.key for p in paramsm.CATALOG if mode == "expert" or p.beginner}
        edits = {k: v for k, v in overrides.items() if k in allowed}
        inputs = {**f["inputs"], **edits}
        P, prov, warnings = paramsm.Resolver(self.sources).resolve(inputs, "expert")
        cal = f.get("soil_calibration")
        if cal:
            soil_changed = any(k in overrides for k in ("sand", "clay", "om", "bd", "soil_type_ua", "lat", "lon"))
            user_theta = any(k in inputs for k in ("theta_fc", "theta_wp"))
            if soil_changed:
                f["soil_calibration"] = None
                warnings.append("Ґрунт змінено — калібрування θFC/θWP за датчиком скинуто; його можна повторити")
            elif not user_theta:
                self._apply_soil_cal(P, prov, cal)
        f.update(inputs=inputs, params=P, provenance=prov, warnings=warnings, mode=mode,
                 soil_hydraulics=WaterBalanceEngine(P).summary())
        # зміна ґрунту/культури робить старий стан некоректним → перерахунок з нуля
        if any(k in overrides for k in ("crop", "sowing_date", "sand", "clay", "om", "bd", "soil_type_ua",
                                        "theta_fc", "theta_wp", "lat", "lon", "stones", "ds", "ds_max",
                                        "sensor_calib", "sensor_offset", "sensor_calibrated", "sensor_depth_cm",
                                        "zr_max_override", "dev_mode", "prev_crop")):
            f["state"] = None
            self._reset_event_flags(f)
            f["ml_local"] = None
            f["ml_log"] = []
            f["last_recommendation"] = None
        if any(k in overrides for k in ("lat", "lon")):
            f.pop("era5l_clim", None)

        self._save(f)
        return dict(field_id=fid, params=P, provenance=prov, warnings=warnings)

    @staticmethod
    def _reset_event_flags(f):
        """Після скидання стану історія відтворюється заново: покази датчика і поливи мають
        увійти в неї знову (інакше вони позначені «використаними» і ігноруються)."""
        f.pop("state_source", None)
        for e in f["events"]["sensor"]:
            e.pop("used", None)
            e.pop("assimilation", None)
        for e in f["events"]["irrigation"]:
            e.pop("applied", None)

    # ------------------------------------------------------------------ калібрування ґрунту
    @staticmethod
    def _apply_soil_cal(P, prov, cal):
        for k in ("theta_fc", "theta_wp"):
            P[k] = round(float(cal[k]), 4)
            prov[k] = dict(value=P[k], source="sensor", symbol="AWC",
                           note=f"калібровано за датчиком ({cal.get('n_events', 0)} подій зволоження, "
                                f"{cal.get('n_days', 0)} діб)", name_uk=prov.get(k, {}).get("name_uk", k))

    @_locked
    def calibrate_soil(self, fid, weather=None, apply=True):
        """
        Калібрує θFC і θWP за всіма показами датчика поля (див. soil_calibration.py).
        weather — необов'язково, dict з "times" і "precipitation" (як у recommend): з ним точніше
        відкидаються події, перервані новим дощем. Записані поливи враховуються як «дощ».
        Якщо θFC/θWP задав сам користувач (режим експерта), вони не змінюються — повертається лише оцінка.
        Після зміни констант стан поля перераховується з історії (як при зміні ґрунту).
        """
        f = self.load_field(fid)
        P = f["params"]
        ev = sorted(f["events"]["sensor"], key=lambda e: e["time"])
        reads = self._sensor_readings(f, P)          # 1.10: профіль кореневої зони, а не змішані глибини
        times = [t for t, _ in reads]
        theta = [v for _, v in reads]
        prior = (f.get("soil_calibration") or {}).get("prior")
        if prior is None:
            P0 = {k: v for k, v in P.items() if k not in ("theta_fc", "theta_wp")}
            for k in ("theta_fc", "theta_wp"):
                if k in f["inputs"]:
                    P0[k] = P[k]
            hyd = WaterBalanceEngine(P0).hyd
            prior = dict(theta_fc=float(hyd["theta_fc"]), theta_wp=float(hyd["theta_wp"]))
        rt = rv = None
        if weather is not None and "precipitation" in weather:
            rt = list(weather["times"])
            rv = list(np.nan_to_num(np.asarray(weather["precipitation"], float)))
        irr = [(_utc(e["time"]), float(e["gross_mm"])) for e in f["events"]["irrigation"]]
        if irr:
            rt = (rt or []) + [t for t, _ in irr]
            rv = (rv or []) + [v for _, v in irr]
        res = estimate_fc_wp(times, theta, prior["theta_fc"], prior["theta_wp"], rt, rv) if len(ev) >= 2 else \
            dict(theta_fc=prior["theta_fc"], theta_wp=prior["theta_wp"], n_events=0, n_days=0,
                 notes=["замало показів датчика (потрібно щонайменше 2, а для оцінки θFC — події зволоження)"])
        res["prior"] = prior
        res["time"] = datetime.now(timezone.utc).isoformat()
        user_set = any(k in f["inputs"] for k in ("theta_fc", "theta_wp"))
        old = (float(WaterBalanceEngine(P).hyd["theta_fc"]), float(WaterBalanceEngine(P).hyd["theta_wp"]))
        changed = abs(res["theta_fc"] - old[0]) > 0.005 or abs(res["theta_wp"] - old[1]) > 0.005
        res["applied"] = bool(apply and changed and not user_set)
        if user_set:
            res["notes"].append("θFC/θWP задано користувачем — калібрування лише показує оцінку, не змінює їх")
        if res["applied"]:
            f["soil_calibration"] = res
            self._apply_soil_cal(P, f["provenance"], res)
            f["soil_hydraulics"] = WaterBalanceEngine(P).summary()
            f["state"] = None                     # розмір резервуара змінився → перерахунок історії
            self._reset_event_flags(f)
        self._save(f)
        return _jsonable(dict(field_id=fid, **res))

    def _replay_for_event(self, f, t):
        """Late observations must be replayed, never silently discarded or double-counted."""
        if f.get("state") and t <= _utc(f["state"]["time"]):
            f["state"] = None
            self._reset_event_flags(f)

    # ------------------------------------------------------------------ події
    @_locked
    def add_sensor_reading(self, fid, theta, depth_cm=None, time=None, sensor_id="s1"):
        f = self.load_field(fid)
        t = _utc(time)
        theta = _number(theta, "theta", maximum=100)
        depth_cm = _number(depth_cm if depth_cm is not None else f["params"].get("sensor_depth_cm", 20),
                           "depth_cm", minimum=2, maximum=150)
        self._replay_for_event(f, t)
        f["events"]["sensor"].append(dict(time=t.isoformat(), theta=theta,
                                          depth_cm=depth_cm,
                                          sensor_id=sensor_id))
        self._save(f)

    @_locked
    def add_fertilizer(self, fid, dose_kg_ha, fert_type="npk_complex", time=None):
        f = self.load_field(fid)
        from .databases import FERTILIZERS
        if fert_type not in FERTILIZERS:
            raise ValueError("Невідомий тип добрива")
        dose_kg_ha = _number(dose_kg_ha, "dose_kg_ha")
        t = _utc(time)
        self._replay_for_event(f, t)
        f["events"]["fertilizer"].append(dict(time=t.isoformat(), dose_kg_ha=dose_kg_ha,
                                              type=fert_type))
        self._save(f)

    @_locked
    def log_irrigation(self, fid, amount_mm=None, time=None, recommended_mm=None, volume_m3=None, note=None,
                       cycles=None):
        """
        Факт поливу (брутто, мм = л/м²). Якщо задано обсяг у м³ — переводиться через площу.
        Порівняння з останньою рекомендацією → навчання на правках користувача.
        cycles (1.10): чи поливали циклами з паузами. None — якщо остання рекомендація (≤ 36 год) радила
        цикли, вважаємо, що порадою скористались; True — уся вода вбралась; False — безперервно (Ksat / I).
        """
        f = self.load_field(fid)
        t = _utc(time)
        if amount_mm is not None and volume_m3 is not None:
            raise ValueError("Задайте лише amount_mm або volume_m3")
        if volume_m3 is not None:
            volume_m3 = _number(volume_m3, "volume_m3", positive=True)
        if recommended_mm is not None:
            recommended_mm = _number(recommended_mm, "recommended_mm", positive=True)
        if cycles is not None and not isinstance(cycles, bool):
            raise ValueError("cycles: потрібне true або false")
        if amount_mm is None and volume_m3 is not None and f["params"].get("area_m2"):
            amount_mm = volume_m3 * 1000.0 / float(f["params"]["area_m2"])
        if amount_mm is None:
            raise ValueError("Потрібна норма поливу (amount_mm) або обсяг (volume_m3) + площа поля")
        amount_mm = _number(amount_mm, "amount_mm", positive=True)
        user = UserPreferenceLearner(self.fe)
        user.load(f["learners"].get("user"))
        last = f.get("last_recommendation")
        learned = None
        if last:
            age_h = (t - _utc(last["time"])).total_seconds() / 3600.0
            if 0 <= age_h <= 36:
                if last["action"] in ("irrigate_now", "postpone") and last.get("gross_mm"):
                    shown_mm = recommended_mm if recommended_mm is not None else last["gross_mm"]
                    # базова норма моделі = показана ÷ навчений множник, що діяв у момент рекомендації
                    # (старі записи без user_mult → 1.0, як і було)
                    mult_then = float(last.get("user_mult") or 1.0)
                    base_mm = shown_mm / max(mult_then, 1e-6)
                    err = user.observe_amount(last["ctx"], base_mm, amount_mm)
                    learned = dict(type="amount", recommended=shown_mm, base=round(base_mm, 2), applied=amount_mm,
                                   error=err)
                    append_training_row(os.path.join(self.data_dir, "training_log.csv"),
                                        self.fe.as_dict(self.fe.build(last["ctx"])),
                                        dict(amount_logratio=math.log(max(amount_mm, 0.1) / base_mm)),
                                        dict(field_id=fid, time=t.isoformat(), shown_mm=shown_mm,
                                             user_mult=mult_then))
                elif last["action"] == "skip":
                    user.observe_timing(last["ctx"]["dr"], last["ctx"]["raw"])
                    learned = dict(type="timing", mad_factor=user.mad_factor)
        f["learners"]["user"] = user.to_dict()
        last = f.get("last_recommendation") or {}
        if cycles is None and last.get("cycles") and abs((t - _utc(last["time"])).total_seconds()) <= 36 * 3600:
            cycles = True
        ev = dict(time=t.isoformat(), gross_mm=float(amount_mm), recommended_mm=recommended_mm, note=note)
        if cycles is not None:
            ev["cycles"] = bool(cycles)
        self._replay_for_event(f, t)
        f["events"]["irrigation"].append(ev)
        self._save(f)
        return dict(ok=True, learned=learned)

    # ------------------------------------------------------------------ погода
    def _weather(self, P, start, end_fc_h, weather=None):
        if weather is not None:
            return weather
        now = datetime.now(timezone.utc)
        past_days = max(1, math.ceil((now - start).total_seconds() / 86400.0) + 1)
        fc_days = max(2, math.ceil(end_fc_h / 24.0) + 1)
        models = None if P.get("weather_model") in (None, "best_match") else P["weather_model"]
        wx = self.sources.forecast(P["lat"], P["lon"], past_days=min(past_days, 92), forecast_days=fc_days,
                                   models=models)
        if past_days > 92:  # довша історія — з архіву ERA5
            try:
                arc = self.sources.archive(P["lat"], P["lon"], start.date(), (wx["times"][0] - timedelta(days=1)).date())
                wx = _concat_weather(arc, wx)
            except SourceError:
                pass
        return wx

    # ------------------------------------------------------------------ головне
    # ------------------------------------------------------------------ ML-ядро прогнозу
    def _sensor_series(self, f, P, until=None):
        """Покази за глибинами {глибина_м: [(час, θ м³/м³)]} з калібруванням фільтра (θ = a + b·показ − зсув)."""
        off = float(P.get("sensor_offset", 0.0))
        cal = P.get("sensor_calib") or [0.0, 1.0]
        out = {}
        for e in f["events"]["sensor"]:
            if until is not None and _utc(e["time"]) > until:
                continue
            th = float(e["theta"])
            th = th / 100.0 if th > 1.0 else th
            d = round(float(e.get("depth_cm") or 20.0) / 100.0, 3)
            out.setdefault(d, []).append((_utc(e["time"]), float(cal[0]) + float(cal[1]) * th - off))
        return {d: sorted(v) for d, v in out.items()}

    def _root_depth(self, f):
        st = f.get("state") or {}
        if st.get("zr"):
            return float(st["zr"])
        return float(WaterBalanceEngine(f["params"]).crop["zr"][1])

    def _sensor_readings(self, f, P, zr=None, until=None):
        """
        1.10: один узгоджений профіль кореневої зони з усіх датчиків (зважено за товщиною шарів),
        а не змішаний ряд різних глибин. zr — глибина коренів (м), за замовчуванням — поточна з стану поля.
        """
        return mlf.build_profile(self._sensor_series(f, P, until=until), zr if zr is not None else self._root_depth(f))

    def _irr_times(self, f):
        return sorted(_utc(e["time"]) for e in f["events"]["irrigation"])

    def _weather_bundle(self, wx_full, P):
        tf = wx_full["times"]
        Wd = {k: np.asarray(v, float) for k, v in wx_full.items()
              if k != "times" and isinstance(v, (list, np.ndarray)) and len(v) == len(tf)}
        _fill_weather_gaps(Wd)
        # ETo для ознак ML — FAO-56, як в Open-Meteo, на якому вчилось ядро (не ASCE, що діє в балансі з 1.2)
        met = meteo.compute_local_eto_series(tf, Wd, P["lat"], P["lon"], P["alt"], {}, method="fao56")
        return dict(times=tf, precip=Wd.get("precipitation"), eto=met["eto_station"], temp=Wd.get("temperature_2m"),
                    rh=Wd.get("relative_humidity_2m"), sw=Wd.get("shortwave_radiation"), wind=Wd.get("wind_speed_10m"))

    def _era5l_state(self, f, P, target, now):
        """Відносна вологість шару target з ERA5-Land у точці: кліматологія (кеш у полі) + останні доби."""
        mix = {"T2": (7, 21, 2), "T3": (7, 21, 27), "RZ": (7, 21, 72)}[target]
        comb = lambda d: (mix[0] * d["sm1"] + mix[1] * d["sm2"] + mix[2] * d["sm3"]) / sum(mix)
        clim = (f.get("era5l_clim") or {}).get(target)
        if clim is None:
            y = now.year
            d = self.sources.era5land_soil(P["lat"], P["lon"], date(y - 3, 3, 20), date(y - 1, 11, 5))
            m = np.array([t.month for t in d["times"]])
            v = comb(d)[(m >= 4) & (m <= 10)]
            v = v[np.isfinite(v)]
            clim = [float(np.percentile(v, 1)), float(np.percentile(v, 99))]
            f.setdefault("era5l_clim", {})[target] = clim
        d = self.sources.era5land_soil(P["lat"], P["lon"], (now - timedelta(days=16)).date(), now.date())
        v = comb(d)
        series = [(t, float(x)) for t, x in zip(d["times"], v) if t.hour == 0 and np.isfinite(x)]
        if not series:
            raise SourceError("ERA5-Land: немає свіжих даних")
        rel = lambda x: (x - clim[0]) / max(clim[1] - clim[0], 1e-6)
        return [(t, rel(x)) for t, x in series], clim

    def _state_context(self, f, P, eng, st_dec, kal, wxb, now, fresh_h, sensor_fault):
        """
        Звідки береться поточний стан ґрунту (див. CHANGELOG 1.7):
          sensor        — свіжий показ датчика (≤ 48 год): фільтр Калмана, як у версії 1.0;
          sensor_stale  — останній показ 2–10 діб тому: він прокручується до сьогодні ML-ядром і подається
                          у фільтр як «слабкий вимір» (похибка росте з віком показу);
          reanalysis    — датчика немає або показ старіший: ERA5-Land у точці (затримка ~5 діб), прокручений
                          ML-ядром до сьогодні, — «слабкий вимір» із великою похибкою;
          model_only    — нічого з наведеного недоступно: фізика і сезонне припущення, як у версії 1.0.
        """
        zr = float(st_dec["zr"])
        fc, wp = eng.hyd["theta_fc"], eng.hyd["theta_wp"]
        b = kal.bias
        taw = float(eng.taw(zr))
        target = mlf.choose_target(zr)
        reads = self._sensor_readings(f, P, zr, until=now)
        irr_t = self._irr_times(f)
        ctx = dict(target=target, zr=zr, taw=taw)
        choice = P.get("forecast_core") or "auto"
        inside, near = mlf.in_domain(float(P["lat"]), float(P["lon"]))
        ctx["domain"] = dict(in_ukraine=inside, nearest_train_km=round(near))
        off = None
        if choice == "physics":
            off = ("user", "вибрано в параметрах: лише фізична модель")
        elif self.ml_core is None:
            off = ("unavailable", self._ml_core_note)
        elif choice == "auto" and not inside:
            off = ("domain", "поле поза Україною — ML-ядро навчене на 14 точках України, тому прогноз робить фізика "
                             "(у режимі експерта ML можна ввімкнути примусово)")
        if off is not None:
            ctx.update(mode="sensor" if (not sensor_fault and fresh_h is not None and 0 <= fresh_h <= 48) else "model_only", ml=False,
                       off_cause=off[0], reason=off[1])
            return ctx
        if not inside or near > mlf.FAR_KM:
            ctx["domain_note"] = (f"поле поза умовами навчання ML-ядра ({round(near)} км від найближчої з 14 точок "
                                  f"навчання) — точність може бути нижчою")
        if reads and not sensor_fault and fresh_h is not None and fresh_h <= 24 * mlf.STALE_MAX_DAYS:
            lo, hi, src = mlf.sensor_scale([t for t, _ in reads], [v for _, v in reads], wp + b, fc + b)
            rel = lambda th: (th - lo) / (hi - lo) if th is not None else None
            ctx.update(ml=True, scale=dict(theta_low=round(lo, 4), theta_high=round(hi, 4), source=src),
                       scale_mm=(hi - lo) * 1000.0 * zr,
                       w_from_dr=lambda dr: (fc - dr / (1000.0 * zr) + b - lo) / (hi - lo),
                       dr_from_w=lambda w: 1000.0 * zr * (fc + b - (lo + w * (hi - lo))))
            if fresh_h <= 48:
                # 1.10: якщо в інтервалі був полив, «зміна за 1/3 доби» містить воду поливу, а не природну динаміку —
                # таку ознаку не подаємо (модель навчена на природних змінах)
                w24 = None if mlf.irrigated_between(irr_t, now - timedelta(hours=24), now) else \
                    rel(mlf.nearest_reading(reads, now - timedelta(hours=24)))
                w72 = None if mlf.irrigated_between(irr_t, now - timedelta(hours=72), now) else \
                    rel(mlf.nearest_reading(reads, now - timedelta(hours=72)))
                ctx.update(mode="sensor", w24=w24, w72=w72)
                return ctx
            t_last, th_last = reads[-1]
            lag = (now - t_last).total_seconds() / 86400.0
            w_now, hist = mlf.rollout(self.ml_core, target, t_last, rel(th_last),
                                      rel(mlf.nearest_reading(reads, t_last - timedelta(hours=24))),
                                      rel(mlf.nearest_reading(reads, t_last - timedelta(hours=72))), now, wxb)
            sig = mlf.sigma_stale(lag)
            ctx.update(mode="sensor_stale", lag_days=round(lag, 1), w_obs=w_now, hist=hist)
            if mlf.irrigated_between(irr_t, t_last - timedelta(hours=78), now):
                ctx["ml"] = False
                # 1.10: після показу був полив — прокрутка ML (природна динаміка) його не знає; стан веде фізика
                ctx["note"] = "після останнього показу був полив — стан веде фізика з фільтром Калмана"
            else:
                ctx.update(dr_obs=ctx["dr_from_w"](w_now), sigma_mm=sig * ctx["scale_mm"])
            return ctx
        try:
            series, clim = self._era5l_state(f, P, target, now)
        except Exception as e:
            ctx.update(mode="model_only", ml=False,
                       reason=f"немає датчика, ERA5-Land недоступний ({str(e)[:60]}) — лише фізика і сезонне припущення")
            return ctx
        if mlf.irrigated_between(irr_t, now - timedelta(days=30), now):
            # 1.10: реаналіз описує богарне поле і не знає поливу — на поливаному полі він тягнув би стан до сухого
            ctx.update(mode="model_only", ml=True, scale=None, scale_mm=taw, w_from_dr=lambda dr: 1.0 - dr / taw,
                       dr_from_w=lambda w: (1.0 - w) * taw,
                       reason="на полі були поливи за 30 діб, яких реаналіз ERA5-Land не бачить — стан веде фізика")
            return ctx
        t_e, w_e = series[-1]
        get = lambda tt: next((w for t, w in series if abs((t - tt).total_seconds()) < 3 * 3600), None)
        w_now, hist = mlf.rollout(self.ml_core, target, t_e, w_e, get(t_e - timedelta(hours=24)),
                                  get(t_e - timedelta(hours=72)), now, wxb)
        lag = (now - t_e).total_seconds() / 86400.0
        ctx.update(mode="reanalysis", ml=True, lag_days=round(lag, 1), w_obs=w_now,
                   scale=dict(theta_low=None, theta_high=None, source=f"кліматологія ERA5-Land у точці {clim}"),
                   scale_mm=taw, w_from_dr=lambda dr: 1.0 - dr / taw, dr_from_w=lambda w: (1.0 - w) * taw,
                   hist=hist)
        ctx["dr_obs"] = ctx["dr_from_w"](w_now)
        ctx["sigma_mm"] = mlf.SIGMA_REANALYSIS * taw
        return ctx

    def _ml_forecast(self, f, P, eng, Sf, st_dec, precip_f, k_learn, wxb, now, ctx):
        """
        Природна (без поливу) зміна вологості на 24/48 год від ML-ядра → зсув траєкторій фізики.
        Повертає опис для відповіді; службові ключі _shift/_shift_sd забирає decide().
        """
        info = dict(core="physics", state_source=ctx["mode"], domain=ctx.get("domain"), off_cause=ctx.get("off_cause"))
        if not ctx.get("ml"):
            info["reason"] = ctx.get("reason") or "ML-ядру потрібен поточний стан (датчик або ERA5-Land)"
            return info
        H = len(Sf.times)
        if H < 48:
            info["reason"] = "прогноз погоди коротший за 48 год"
            return info
        try:
            zr, target = ctx["zr"], ctx["target"]
            w_now = ctx["w_from_dr"](float(st_dec["dr"]))
            hist = ctx.get("hist") or []
            from_hist = lambda dt: next((w for t, w in reversed(hist)
                                         if abs((t - (now - dt)).total_seconds()) < 3 * 3600), None)
            w24 = ctx.get("w24") if ctx["mode"] == "sensor" else from_hist(timedelta(hours=24))
            w72 = ctx.get("w72") if ctx["mode"] == "sensor" else from_hist(timedelta(hours=72))
            wd = mlf.daily_weather(wxb["times"], now, wxb["precip"], wxb["eto"], wxb["temp"], wxb["rh"], wxb["sw"],
                                   wxb["wind"])
            if mlf.irrigated_between(self._irr_times(f), now - timedelta(hours=30), now):
                w24 = None
            if mlf.irrigated_between(self._irr_times(f), now - timedelta(hours=78), now):
                w72 = None
            local = self._load_local(f) if ctx["mode"] in ("sensor", "sensor_stale") else {}
            dw, feats_h = {}, {}
            for h in (1, 2):
                ft = mlf.build_features(now, w_now, w24, w72, wd, h)
                core_p = self.ml_core.predict(target, h, ft)
                ft["core"] = core_p
                if local.get(h) is not None:
                    import pandas as pd
                    cols = mlf.feature_names(h) + ["core"]
                    core_p += float(local[h].predict(pd.DataFrame([{c: ft.get(c, np.nan) for c in cols}]))[0])
                dw[h], feats_h[h] = core_p, ft
            st = eng.init_state(1, float(st_dec["dr"]), x0=st_dec.get("x", 0.0), zr0=zr)
            st["de"] = np.full(1, float(st_dec.get("de", eng.rew)))
            st["fw"] = np.full(1, float(st_dec.get("fw", 1.0)))
            _, rec = eng.run(Sf, st, precip_f, np.zeros(H), learn_k=k_learn, i0=0, i1=48)
            dr_phys = {1: float(rec["dr"][0, 23]), 2: float(rec["dr"][0, 47])}
            scale_mm = ctx["scale_mm"]
            c = {h: (float(st_dec["dr"]) - dw[h] * scale_mm) - dr_phys[h] for h in (1, 2)}
            sd = {h: self.ml_core.sigma(target, h) * scale_mm for h in (1, 2)}
            mm2th = lambda mm: round(mm / (1000.0 * zr), 4)
            info = dict(core="ml" + ("+local" if any(local.get(h) is not None for h in (1, 2)) else ""),
                        state_source=ctx["mode"], note=self._ml_core_note, target=target, scale=ctx.get("scale"),
                        domain=ctx.get("domain"), domain_note=ctx.get("domain_note"),
                        state_note=ctx.get("note") or (ctx.get("reason") if ctx.get("mode") == "model_only" else None),
                        w_now=round(w_now, 3),
                        dtheta_ml={f"{24 * h}h": mm2th(dw[h] * scale_mm) for h in (1, 2)},
                        dtheta_physics={f"{24 * h}h": mm2th(float(st_dec["dr"]) - dr_phys[h]) for h in (1, 2)},
                        shift_mm={f"{24 * h}h": round(c[h], 2) for h in (1, 2)},
                        sigma_mm={f"{24 * h}h": round(sd[h], 2) for h in (1, 2)},
                        _shift=mlf.shift_profile(H, c[1], c[2]), _shift_sd=mlf.sd_profile(H, sd[1], sd[2]))
            if ctx["mode"] in ("sensor", "sensor_stale"):     # журнал для донавчання — лише з даними датчика
                info["_log"] = dict(target=target, lo=ctx["scale"]["theta_low"], hi=ctx["scale"]["theta_high"],
                                    w_now=w_now, zr=round(zr, 3),
                                    feats={str(h): {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                                                    for k, v in feats_h[h].items()} for h in (1, 2)})
        except Exception as e:                       # ML-ядро не повинно ламати рекомендацію
            info = dict(core="physics", state_source=ctx["mode"],
                        reason=f"помилка ML-ядра ({type(e).__name__}: {str(e)[:80]})")
        return info

    def _ml_log(self, f, core, now):
        """Раз на ~добу зберігаємо ознаки ML для майбутнього локального донавчання (мітки — з показів датчика)."""
        lg = core.pop("_log", None)
        if lg is None:
            return
        log = f.setdefault("ml_log", [])
        if log and (now - _utc(log[-1]["time"])) < timedelta(hours=20):
            return
        log.append(dict(time=now.isoformat(), **lg))
        if len(log) > 800:
            del log[: len(log) - 800]

    def _local_path(self, fid):
        return os.path.join(self.data_dir, "models", f"{fid}_local.joblib")

    def _load_local(self, f):
        if not (f.get("ml_local") or {}).get("enabled"):
            return {}
        try:
            import joblib
            d = joblib.load(self._local_path(f["field_id"]))
            if d.get("compatibility") != "ml_core_v2/profile_v2/irrigation_guard_v3":
                f["ml_local"]["enabled"] = False
                f["ml_local"]["reason"] = "Несумісна локальна модель; потрібне повторне навчання"
                return {}
            return {int(k): v for k, v in d["models"].items()}
        except Exception:
            return {}

    def _local_rows(self, f, reads, h, until=None):
        """Приклади для локального донавчання: ознаки з журналу + фактична зміна θ за 24·h год із показів."""
        series = self._sensor_series(f, f["params"], until=until)
        irr_t = self._irr_times(f)
        prof_cache = {}
        rows = []
        for r in f.get("ml_log", []):
            t = _utc(r["time"])
            # 1.10: мітка — природна зміна; інтервали з поливом (із запасом 2 год) не беремо
            if mlf.irrigated_between(irr_t, t - timedelta(hours=7), t + timedelta(hours=24 * h + 7)):
                continue
            zr = round(float(r.get("zr") or self._root_depth(f)), 2)   # профіль тієї ж глибини, що й у прогнозі
            if zr not in prof_cache:
                pr_ = mlf.build_profile(series, zr)
                prof_cache[zr] = (pr_, [x[0] for x in pr_])
            reads, times = prof_cache[zr]
            th1 = mlf.nearest_reading(reads, t + timedelta(hours=24 * h), times=times)
            th0 = mlf.nearest_reading(reads, t, times=times)
            if th1 is None or th0 is None:
                continue
            ft = {k: (np.nan if v is None else v) for k, v in r["feats"][str(h)].items()}
            rows.append(dict(time=t, y=(th1 - th0) / (r["hi"] - r["lo"]), **ft))
        return rows

    def _finetune_on(self, f, now=None, auto=False):
        """Донавчання на полі f у пам'яті (збереження — справа того, хто викликав)."""
        reads = None
        models, infos, n = {}, {}, 0
        for h in (1, 2):
            rows = self._local_rows(f, reads, h, until=now or datetime.now(timezone.utc))
            n = max(n, len(rows))
            m, info = mlf.LocalResidual.fit(rows, h)
            infos[str(h)] = info
            if m is not None:
                models[h] = m
        enabled = bool(models)
        if enabled:
            import joblib
            os.makedirs(os.path.dirname(self._local_path(f["field_id"])), exist_ok=True)
            joblib.dump(dict(compatibility="ml_core_v2/profile_v2/irrigation_guard_v3", models=models),
                        self._local_path(f["field_id"]), compress=3)
        f["ml_local"] = dict(enabled=enabled, horizons=infos, n_rows=n, auto=auto,
                             time=(now or datetime.now(timezone.utc)).isoformat())
        return f["ml_local"]

    @_locked
    def finetune_local(self, fid):
        """
        Локальне донавчання ML-ядра на даних цього поля вручну (режим експерта). З версії 1.9 воно також
        запускається автоматично, коли накопичиться ≥ 60 діб показів (див. _maybe_auto_finetune).
        Мітки — фактична зміна θ за 24/48 год з показів датчика (±3 год), у масштабі, який діяв у момент прогнозу.
        Поправка вмикається лише якщо на останніх 25 % прикладів зменшує похибку ядра.
        """
        f = self.load_field(fid)
        if self.ml_core is None:
            return dict(field_id=fid, enabled=False, reason=self._ml_core_note)
        res = self._finetune_on(f, auto=False)
        self._save(f)
        return _jsonable(dict(field_id=fid, **res))

    def _maybe_auto_finetune(self, f, now):
        """
        Автоматичне донавчання (1.9): коли є ≥ 60 діб прикладів із мітками. Повторно — через 30 діб,
        якщо з'явились нові приклади (і після невдалої спроби, і для оновлення вже ввімкненої поправки).
        """
        if self.ml_core is None or not f.get("ml_log"):
            return None
        n = len(self._local_rows(f, None, 1, until=now))
        f["ml_progress"] = dict(rows=n, need=mlf.LOCAL_MIN_ROWS)
        if n < mlf.LOCAL_MIN_ROWS:
            return None
        last = f.get("ml_local")
        if last:
            age = (now - _utc(last["time"])).total_seconds() / 86400.0
            if age < mlf.LOCAL_RETRY_DAYS or n <= int(last.get("n_rows", 0)):
                return None
        return self._finetune_on(f, now=now, auto=True)

    @staticmethod
    def _guidance(f, core, auto_ft=None):
        """Панель «Як система рахує це поле»: поточний режим і один конкретний наступний крок (1.9)."""
        src, cr = core.get("state_source"), core.get("core", "physics")
        lag = (core.get("state") or {}).get("lag_days")
        note = core.get("domain_note")
        prog = f.get("ml_progress") or {}
        loc = f.get("ml_local") or {}
        def gain(l):
            h = (l.get("horizons") or {}).get("1") or {}
            if h.get("mae_core_val"):
                return round(100 * (1 - h["mae_local_val"] / h["mae_core_val"]))
            return None
        if src in ("reanalysis", "model_only"):
            g = dict(stage="no_sensor", title="Поле без датчика",
                     message=("Стан ґрунту оцінено без вимірювань " +
                              ("(реаналіз ERA5-Land, прокручений ML-моделлю)" if src == "reanalysis"
                               else "(фізична модель з урахуванням записаних поливів)" if core.get("state_note")
                               else "(лише фізична модель і сезонне припущення)") +
                              " — рекомендація орієнтовна."),
                     next_step=("Додайте показ датчика або ручний вимір вологості: на перевірці частка правильних "
                                "рішень «сухо / волого» зросла з ~75% без датчика до 87–94% з показом 3–7-денної давності."))
        elif src == "sensor_stale":
            g = dict(stage="stale_sensor", title="Показ датчика застарів",
                     message=f"Останній показ — ≈ {lag:.0f} діб тому; стан прокручено до сьогодні ML-моделлю.",
                     next_step="Додайте свіжий показ: з показом до 3 діб точність рішень ~92–94%, до 7 діб — ~87–89%.")
        elif cr == "physics":
            cause = core.get("off_cause")
            msg = {"domain": "Поле поза Україною: ML-ядро навчене на 14 точках України, тому прогноз робить фізика.",
                   "user": "У параметрах обрано «лише фізична модель».",
                   "unavailable": "ML-ядро недоступне — прогноз робить фізика."}.get(cause, "Прогноз робить фізика.")
            nxt = {"domain": "Нічого не потрібно. Якщо умови поля схожі на українські, ML можна ввімкнути примусово "
                             "в режимі експерта (параметр «Ядро прогнозу»).",
                   "user": "На всіх перевірках ML точніша за фізику; поверніть «Автоматично», якщо немає особливих причин.",
                   "unavailable": core.get("reason") or "Перевірте встановлення scikit-learn і файлів моделей."}.get(
                cause, "—")
            g = dict(stage="physics_only", title="Прогноз — фізична модель", message=msg, next_step=nxt)
        elif loc.get("enabled"):
            pct = gain(loc)
            g = dict(stage="local", title="Модель донавчена на вашому полі",
                     message=("Прогноз — ML-модель, донавчена на даних цього поля" +
                              (f": на відкладених даних похибка менша на {pct}%." if pct is not None else ".")),
                     next_step="Нічого не потрібно: модель оновлюється автоматично раз на 30 діб нових даних.")
        elif loc and not loc.get("enabled") and int(loc.get("n_rows", 0)) >= mlf.LOCAL_MIN_ROWS:
            g = dict(stage="local_rejected", title="Донавчання перевірено",
                     message="Донавчання на вашому полі поки не покращило прогноз — використовується загальна модель.",
                     next_step="Нічого не потрібно: система повторить перевірку через ~30 діб нових даних.")
        else:
            n, need = int(prog.get("rows", 0)), int(prog.get("need", mlf.LOCAL_MIN_ROWS))
            g = dict(stage="learning", title="Збираємо дані поля",
                     message=f"Прогноз — загальна ML-модель (навчена на даних України). Даних поля для донавчання: "
                             f"{n} із {need} діб.",
                     next_step=(f"Ще ≈ {max(need - n, 0)} діб регулярних показів (щодня) — і модель автоматично "
                                f"донавчиться на вашому полі (на перевірці це зменшувало похибку ще на ~10–20%)."))
        if auto_ft is not None:
            g["event"] = ("Щойно автоматично донавчено модель на даних поля." if auto_ft.get("enabled") else
                          "Щойно перевірено донавчання на даних поля: поки без покращення.")
        if note:
            g["message"] += " Увага: " + note + "."
        if core.get("state_note") and src != "model_only":
            g["message"] += " " + core["state_note"][0].upper() + core["state_note"][1:] + "."
        return g

    def _measurement_required(self, f, fid, now, tz, new_state, core, notes):
        """Політика «off»: без датчика рекомендацію не видаємо — просимо зробити вимір."""
        f["state"] = new_state
        f["last_recommendation"] = dict(time=now.isoformat(), action="measure", gross_mm=None)
        self._save(f)
        msg = ("Для цього поля немає показів датчика. Рекомендації без вимірювання вимкнено налаштуванням "
               "застосунку — додайте показ датчика або ручний вимір вологості.")
        return _jsonable(dict(field_id=fid, time_utc=now.isoformat(),
                              time_local=(now + timedelta(hours=tz)).strftime("%Y-%m-%d %H:%M"),
                              action="measure", action_uk=explain.ACTION_UK["measure"],
                              headline=explain.ACTION_UK["measure"], summary=explain.ACTION_UK["measure"] + ". " + msg,
                              details=[msg], amount=None, confidence=0.0, confidence_label="—",
                              forecast_core=core, orientative=False, guidance=self._guidance(f, core),
                              notes=notes + f.get("warnings", []),
                              reasons=[dict(code="measurement_required")]))

    # ------------------------------------------------------------------ головне
    @_locked
    def recommend(self, fid, now=None, weather=None, ensemble=None):
        f = self.load_field(fid)
        P = dict(f["params"])
        user = UserPreferenceLearner(self.fe)
        user.load(f["learners"].get("user"))
        P["mad_factor"] = user.mad_factor
        eng = WaterBalanceEngine(P)
        crop = eng.crop
        live = now is None
        now = _utc(now)
        if live and (now.minute or now.second):
            # поточна година вже йде: включаємо її, щоб щойно введені показ датчика / полив урахувалися
            now = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            now = now.replace(minute=0, second=0, microsecond=0)
        horizon = int(P.get("horizon_h", 72))
        sowing = _utc(P["sowing_date"])
        st = f.get("state")
        if st and _utc(st["time"]) > now:
            raise ValueError("Час рекомендації раніший за збережений стан поля")
        if st:
            start = _utc(st["time"])
        else:
            start = max(sowing - timedelta(days=1), now - timedelta(days=60))
            if sowing > now:
                start = now - timedelta(days=3)
        notes = []
        wx = self._weather(P, min(start, now - timedelta(days=18)), horizon, weather)
        wx_full = wx
        times = wx["times"]
        if not times or any(b - a != timedelta(hours=1) for a, b in zip(times, times[1:])):
            raise ValueError("Погода має містити непорожній, послідовний погодинний ряд")
        if not any(t <= now for t in times) or times[-1] < now + timedelta(hours=6):
            raise ValueError("Ряд погоди не покриває поточний час і щонайменше 6 год прогнозу")
        # обрізаємо ряд: [start, now + horizon]
        # годину стану теж беремо (повторний виклик у ту саму годину не повинен давати порожній ряд)
        i_a = next((i for i, t in enumerate(times) if t > start - timedelta(hours=1)), 0)
        i_b = next((i for i, t in enumerate(times) if t > now + timedelta(hours=horizon)), len(times))
        times = times[i_a:i_b]
        if not times or not any(t <= now for t in times):
            raise ValueError("Ряд погоди не покриває стан поля")
        W = {k: (np.asarray(v)[i_a:i_b] if isinstance(v, (list, np.ndarray)) and len(v) == len(wx["times"]) else v)
             for k, v in wx.items() if k != "times"}
        i_now = max(0, max(i for i, t in enumerate(times) if t <= now))
        if times[-1] < now + timedelta(hours=24):
            notes.append("Прогноз погоди коротший за 24 год — рішення менш надійне")
        for key in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation", "precipitation"):
            try:
                values = np.asarray(W[key], float)
                valid = values.ndim == 1 and len(values) == len(times) and np.isfinite(values).any()
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"Погода: немає придатного погодинного ряду {key}")
        _fill_weather_gaps(W)
        tz = P.get("tz_offset_h") if P.get("tz_offset_h") is not None else _tz_offset(now)
        # ---- стадія розвитку
        mode = P.get("dev_mode", "thermal")
        if st and st.get("dvs") is not None:
            dvs = cropm.development_series(times, W["temperature_2m"], sowing, crop, mode,
                                           dvs0=st["dvs"], t0=_utc(st["time"]))
        else:
            dvs0 = None
            if sowing < times[0]:  # частина сезону поза рядом погоди — оцінка за календарем
                dvs0 = (times[0] - sowing).total_seconds() / 86400.0 / float(sum(crop["stages"]))
            dvs = cropm.development_series(times, W["temperature_2m"], sowing, crop, mode,
                                           dvs0=dvs0, t0=times[0] if dvs0 is not None else None)
            if dvs0 is not None and mode == "thermal" and not crop.get("winter"):
                dvs = np.maximum(dvs, dvs0)
        fert = [dict(e, time=_utc(e["time"])) for e in f["events"]["fertilizer"] if e.get("time")]
        S = eng.prepare(times, W, dvs, _soil_temp(W), fert, tz)
        T = len(times)
        # ---- історичні опади та поливи
        rain_hist = np.nan_to_num(np.asarray(W.get("precipitation", np.zeros(T)), float))
        irr = np.zeros(T)
        irr_infil = np.full(T, eng.infiltration_fraction())     # 1.10: полив циклами вбирається повністю
        i_start = 0
        if st:
            i_start = next((i for i, t in enumerate(times) if t > _utc(st["time"])), 0)
        for e in f["events"]["irrigation"]:
            if e.get("applied"):
                continue
            if e.get("cycles") and eng.infiltration_fraction() < 1.0:
                ks = eng.hyd["ksat_mm_h"]
                begin = _utc(e["time"])
                end = begin + timedelta(hours=e["gross_mm"] / ks)
                for jj in range(i_start, i_now + 1):
                    overlap = (min(times[jj], end) - max(times[jj] - timedelta(hours=1), begin)).total_seconds() / 3600
                    if overlap > 0:
                        irr[jj] += overlap * ks
                        irr_infil[jj] = 1.0
                if times[i_now] >= end:
                    e["applied"] = True
                continue
            if (times[i_start] - _utc(e["time"])) > timedelta(hours=6):
                e["applied"] = True          # полив до початку відтворення історії — уже «в минулому» стану
                continue
            j = max(_hour_index(times, _utc(e["time"])), i_start)   # запізнілий запис → у перший крок
            if j <= i_now:
                irr[j] += e["gross_mm"]
                if e.get("cycles"):
                    irr_infil[j] = 1.0
                e["applied"] = True
        # ---- стартовий стан
        if st:
            state = dict(dr=np.array([st["dr"]]), de=np.array([st["de"]]), x=np.array([st["x"]]),
                         du=np.array([st.get("du", max(st["dr"], 0.0) * min(0.3, st["zr"]) / max(st["zr"], 1e-6))]),
                         swe=np.array([st.get("swe", 0.0)]),
                         fw=np.array([st.get("fw", 1.0)]), zr=st["zr"], h_since_irr=np.array([st.get("h_since_irr", 999)]))
            kal = SoilMoistureKalman.from_dict(st["kalman"], du_default=float(state["du"][0]))
        else:
            taw0 = eng.taw(S.zr[0])
            dr_prior, sig0, src0 = _initial_depletion(eng, S, W, start, taw0)
            notes.append(src0)
            state = eng.init_state(1, dr_prior, zr0=S.zr[0])
            kal = SoilMoistureKalman(dr_prior, sig0, sigma_bias0=0.008 if P.get("sensor_calibrated") else 0.02,
                                     du0=float(state["du"][0]))
        # ---- показання датчиків по годинах
        obs = {}
        for e in f["events"]["sensor"]:
            if e.get("used"):
                continue
            te = _utc(e["time"])
            j = _hour_index(times, te)
            if j < i_start and (times[i_start] - te) > timedelta(hours=6):
                e["used"] = True          # надто старий показ — у минуле не повертаємось
                continue
            j = max(j, i_start)
            if j <= i_now:
                obs.setdefault(j, []).append(e)
                e["used"] = True
        sens_off = float(P.get("sensor_offset", 0.0))
        cal = P.get("sensor_calib") or [0.0, 1.0]           # θ = a + b·показ (v1.2)
        cal_a, cal_b = float(cal[0]), float(cal[1])
        kal_logs = []

        def cb(i, stt, flux):
            kal.predict(stt["dr"][0], flux, du_model=float(stt["du"][0]))
            if i in obs:
                for e in obs[i]:
                    sat_cap = eng.sat_unit * S.zr[i]
                    th_raw = float(e["theta"])
                    th_raw = th_raw / 100.0 if th_raw > 1.0 else th_raw
                    th = cal_a + cal_b * th_raw - sens_off          # калібрування «показ → θ» (v1.2)
                    layer = _sensor_layer(e.get("depth_cm"), flux, S.zr[i])
                    info = kal.update(th, S.zr[i], eng.hyd["theta_fc"],
                                      P.get("sigma_sens", 0.01), P.get("sigma_spat", 0.015), e.get("depth_cm"),
                                      dr_bounds=(-sat_cap, float(flux["taw"][0])), t=times[i], layer=layer)
                    kal_logs.append(info)
                    e["assimilation"] = info
                    if info["accepted"]:
                        lk[0] = kal.k_field
                stt["dr"][0] = kal.dr
                stt["du"][0] = kal.du

        # поправка транспірації поля оцінюється лише фільтром Калмана (змінна стану k)
        lk = [kal.k_field]
        if i_now >= i_start:
            state, hist = eng.run(S, state, rain_hist, irr, learn_k=lk, i0=i_start, i1=i_now + 1, irr_infil=irr_infil,
                                  callback=cb)
        k_learn = float(np.clip(kal.k_field, 0.6, 1.5))
        recent_sensor_logs = [e["assimilation"] for e in sorted(f["events"]["sensor"], key=lambda e: e["time"])
                              if e.get("assimilation") and _utc(e["time"]) <= now][-3:]
        sensor_fault = any(not l["accepted"] for l in recent_sensor_logs)
        # ---- збереження стану на «зараз»
        new_state = dict(time=times[i_now].isoformat(), dr=float(state["dr"][0]), de=float(state["de"][0]),
                         du=float(state["du"][0]), swe=float(state["swe"][0]),
                         x=float(state["x"][0]), fw=float(state["fw"][0]), zr=float(state["zr"]),
                         h_since_irr=float(state["h_since_irr"][0]), dvs=float(dvs[i_now]), kalman=kal.to_dict())
        # ---- прогноз і рішення
        a, b = i_now + 1, T
        if b - a < 6:
            raise RuntimeError("Немає прогнозу погоди на майбутнє — перевірте з'єднання з Open-Meteo")
        Wf = {k: (v[a:b] if isinstance(v, np.ndarray) and len(v) == T else v) for k, v in W.items()}
        Sf = eng.prepare(times[a:b], Wf, dvs[a:b], _soil_temp(Wf), fert, tz)
        precip_f = np.nan_to_num(np.asarray(Wf.get("precipitation", np.zeros(b - a)), float))
        prob_f = np.asarray(Wf.get("precipitation_probability", np.full(b - a, np.nan)), float)
        members = None
        if ensemble is not None:
            members = _align_members(ensemble, times[a:b])
        elif P.get("use_ensemble", True) and weather is None:
            try:
                ens = self.sources.ensemble_precip(P["lat"], P["lon"], forecast_days=math.ceil(horizon / 24) + 1)
                members = _align_members(ens, times[a:b])
            except Exception as e:
                notes.append(f"Ансамбль опадів недоступний ({str(e)[:60]}) — невизначеність з імовірності опадів")
        ctx_now = _ctx(S, i_now, state, None, P, eng, rain_hist, precip_f, prob_f)
        user_mult = user.multiplier(ctx_now)
        P["user_mult_max"] = user.max_up
        last_sensor = max((_utc(e["time"]) for e in f["events"]["sensor"] if _utc(e["time"]) <= now), default=None)
        fresh_h = (now - last_sensor).total_seconds() / 3600.0 if last_sensor else None
        st_dec = dict(dr=new_state["dr"], de=new_state["de"], x=new_state["x"], fw=new_state["fw"], zr=new_state["zr"],
                      du=new_state["du"], swe=new_state["swe"])
        P["_rain_waits"] = int((st or {}).get("rain_waits", 0))
        if st and _utc(st["time"]) == now and (f.get("last_recommendation") or {}).get("waiting_for_rain"):
            P["_rain_waits"] = max(0, P["_rain_waits"] - 1)
        # ---- поточний стан без свіжого датчика: ERA5-Land або старий показ, прокручені ML-ядром (1.7)
        try:
            wxb = self._weather_bundle(wx_full, P)
            ctx = self._state_context(f, P, eng, st_dec, kal, wxb, now, fresh_h, sensor_fault)
        except Exception as e:
            wxb, ctx = None, dict(mode="model_only", ml=False, reason=f"оцінка стану недоступна ({str(e)[:60]})")
        state_info = dict(mode=ctx["mode"], lag_days=ctx.get("lag_days"))
        if ctx.get("dr_obs") is not None:
            last = (f.get("state_source") or {}).get("applied")
            if last is None or (now - _utc(last)) >= timedelta(hours=20) or ctx["mode"] != f["state_source"].get("mode"):
                info = kal.update_dr(ctx["dr_obs"], ctx["sigma_mm"], dr_bounds=(-eng.sat_unit * ctx["zr"], ctx["taw"]),
                                     source=ctx["mode"], t=now)
                st_dec["dr"] = new_state["dr"] = kal.dr
                st_dec["du"] = new_state["du"] = kal.du
                new_state["kalman"] = kal.to_dict()
                f["state_source"] = dict(mode=ctx["mode"], applied=now.isoformat())
                state_info.update(applied=True, dr_obs_mm=round(ctx["dr_obs"], 1), sigma_mm=round(ctx["sigma_mm"], 1),
                                  gain=round(info["gain"], 2))
            else:
                state_info["applied"] = False
        core = self._ml_forecast(f, P, eng, Sf, st_dec, precip_f, k_learn, wxb, now, ctx) if wxb is not None else \
            dict(core="physics", state_source=ctx["mode"], reason=ctx.get("reason"))
        core["state"] = state_info
        no_sensor = ctx["mode"] in ("reanalysis", "model_only")
        if no_sensor and self.no_sensor_policy == "off":
            return self._measurement_required(f, fid, now, tz, new_state, core, notes)
        dec = decide(eng, Sf, st_dec, kal.sigma_dr, precip_f, prob_f, P, user_mult=user_mult, k_learn=k_learn,
                     members=members, rng=np.random.default_rng(self.seed), sensor_ok=not sensor_fault,
                     sensor_fresh_h=fresh_h, dr_shift=core.pop("_shift", None), dr_shift_sd=core.pop("_shift_sd", None))
        if sensor_fault:
            dec["reasons"].append(dict(code="sensor_fault"))
        if core["state"]["mode"] == "reanalysis":
            dec["reasons"].append(dict(code="state_reanalysis", lag=core["state"].get("lag_days")))
        elif core["state"]["mode"] == "sensor_stale":
            dec["reasons"].append(dict(code="state_stale_sensor", lag=core["state"].get("lag_days")))
        elif fresh_h is None or fresh_h > 48:
            dec["reasons"].append(dict(code="no_sensor"))
        orientative = no_sensor and self.no_sensor_policy == "orientative"
        if orientative:
            _make_orientative(dec, kal.sigma_dr)
        # ---- вологість: модель + датчик
        zr = new_state["zr"]
        theta = eng.theta_from_dr(new_state["dr"], zr)
        moisture = dict(theta=float(theta), theta_sigma=float(kal.sigma_dr / (1000 * zr)),
                        theta_fc=eng.hyd["theta_fc"], theta_wp=eng.hyd["theta_wp"],
                        depletion_mm=round(new_state["dr"], 1), depletion_sigma_mm=round(kal.sigma_dr, 1),
                        taw_mm=round(dec["state"]["taw"], 1), raw_mm=round(dec["state"]["raw"], 1),
                        available_water_pct=round(100 * max(0.0, 1 - new_state["dr"] / max(dec["state"]["taw"], 1e-6)), 1),
                        sensor_bias=round(kal.bias, 4), sensor_bias_sigma=round(kal.sigma_bias, 4),
                        fc_anchors=kal.n_pins, root_depth_m=round(zr, 2),
                        # v1.2: верхній шар 0–0.3 м окремо (його бачить датчик на 10–30 см) і сніговий запас
                        theta_upper=round(float(eng.hyd["theta_fc"] - new_state["du"] / (1000 * min(0.3, zr))), 4),
                        theta_upper_sigma=round(float(kal.sigma_du / (1000 * min(0.3, zr))), 4),
                        snow_swe_mm=round(new_state["swe"], 1))
        if recent_sensor_logs:
            moisture["last_sensor"] = recent_sensor_logs[-1]
        when_local = None
        if dec["action"] == "postpone":
            t_when = Sf.times[dec["t_idx"]] if dec["t_idx"] is not None else now + timedelta(hours=dec["postpone_hours"])
            when_local = (t_when + timedelta(hours=tz)).strftime("%d.%m %H:%M")
        ex = explain.build(dec, moisture, when_local, orientative=orientative)
        # ---- поточні показники
        day_slice = slice(max(0, i_now - 23), i_now + 1)
        if i_now - day_slice.start < 23:   # мало історії (повторний виклик) — беремо найближчі 24 год прогнозу
            day_slice = slice(i_now, min(T, i_now + 24))
        today = dict(eto_mm_day=round(float(S.eto_station[day_slice].sum()), 2),
                     eto_local_mm_day=round(float(S.eto[day_slice].sum()), 2),
                     k_micro=round(float(np.mean(S.k_micro[day_slice][S.is_day[day_slice]])) if S.is_day[day_slice].any() else 1.0, 3),
                     kcb=round(float(S.kcb[i_now]), 3), kc_max=round(float(S.kcmax[i_now]), 3),
                     fc_cover=round(float(S.fc[i_now]), 2), p=round(float(S.p[i_now]), 3),
                     dvs=round(float(dvs[i_now]), 3), stage=cropm.stage_name(float(dvs[i_now]), crop),
                     ece=round(float(eng.ece_base + new_state["x"]), 2), k_learn=round(k_learn, 3),
                     k_field=round(kal.k_field, 3),
                     mad_factor=round(user.mad_factor, 3))
        fc_days = []
        for d0 in range(0, b - a, 24):
            sl = slice(d0, min(d0 + 24, b - a))
            fc_days.append(dict(date=(Sf.times[sl.start] + timedelta(hours=tz)).strftime("%d.%m"),
                                rain_mm=round(float(precip_f[sl].sum()), 1),
                                rain_prob_max=None if np.all(np.isnan(prob_f[sl])) else float(np.nanmax(prob_f[sl])),
                                eto_mm=round(float(Sf.eto[sl].sum()), 2),
                                tmax=round(float(np.max(Sf.t_air[sl])), 1), tmin=round(float(np.min(Sf.t_air[sl])), 1)))
        # ---- збереження
        ctx_now.update(dr=new_state["dr"], raw=dec["state"]["raw"])
        # лічильник «чекаємо дощ» поспіль: росте при кожному такому рішенні, обнуляється при будь-якому іншому
        new_state["rain_waits"] = (P["_rain_waits"] + 1) if dec.get("waiting_for_rain") else 0
        f["state"] = new_state
        f["learners"].pop("residual", None)   # старий учень прибрано; стан у старих файлах ігноруємо
        f["last_recommendation"] = dict(time=now.isoformat(), action=dec["action"],
                                        gross_mm=(dec["amount"] or {}).get("gross_mm"),
                                        cycles=bool((dec["amount"] or {}).get("cycles")),
                                        user_mult=user_mult, waiting_for_rain=bool(dec.get("waiting_for_rain")), ctx=ctx_now)
        self._ml_log(f, core, now)
        auto_ft = None
        try:
            auto_ft = self._maybe_auto_finetune(f, now)
        except Exception as e:                       # донавчання не повинно ламати рекомендацію
            notes.append(f"Автоматичне донавчання не вдалося: {str(e)[:80]}")
        guidance = self._guidance(f, core, auto_ft)
        self._save(f)
        out = dict(field_id=fid, time_utc=now.isoformat(), forecast_core=core, orientative=orientative,
                   guidance=guidance,
                   time_local=(now + timedelta(hours=tz)).strftime("%Y-%m-%d %H:%M"),
                   action=dec["action"], action_uk=explain.ACTION_UK[dec["action"]],
                   headline=ex["headline"], summary=ex["summary"], details=ex["details"],
                   when_local=when_local, postpone_hours=dec["postpone_hours"], amount=dec["amount"],
                   next_check_hours=dec["next_check_hours"], confidence=dec["confidence"],
                   confidence_label=dec["confidence_label"], moisture=moisture, today=today,
                   risk=dec["risk"], rain=dec["rain"], forecast_days=fc_days,
                   trajectory=dec["trajectory"], notes=notes + f.get("warnings", []),
                   reasons=dec["reasons"])
        return _jsonable(out)


# ---------------------------------------------------------------------- режим без датчика (1.8, варіант «Г»)
def _round5(x):
    return float(5.0 * round(float(x) / 5.0))


def _make_orientative(dec, sigma_dr):
    """
    Орієнтовна рекомендація для поля без датчика. Причина (Етап 3, ds/stage3/no_sensor.py): без датчика стан
    оцінюється правильно в ~75–77% випадків, а третина сухих днів пропускається; точна норма в міліметрах
    створювала б хибне враження точності. Тому: впевненість завжди «низька», замість норми — діапазон
    10–90% сценаріїв Монте-Карло (округлено до 5 мм, медіана всередині), і порада зробити вимір.
    Сама дія (поливати / відкласти / ні) не змінюється. sigma_dr лишено в сигнатурі для сумісності.
    """
    dec["confidence"] = min(float(dec.get("confidence", 0.3)), 0.3)
    dec["confidence_label"] = "низька"
    am = dec.get("amount")
    if am:
        g = float(am["gross_mm"])
        p10, p90 = am.get("gross_p10_p90", [g, g])
        lo = 5.0 * math.floor(min(p10, g) / 5.0)
        hi = 5.0 * math.ceil(max(p90, g) / 5.0)
        if hi - lo < 5.0:
            hi = lo + 5.0
        am["range_mm"] = [lo, hi]
        am["orientative"] = True
    dec["reasons"].append(dict(code="orientative_no_sensor"))


# ---------------------------------------------------------------------- допоміжне
def _ctx(S, i, state, flux, P, eng, rain_hist, precip_f=None, prob_f=None):
    i = min(i, len(S.times) - 1)
    lo = max(0, i - 72)
    taw = eng.taw(S.zr[i])
    t = S.times[i]
    ctx = dict(dr_rel=float(state["dr"][0] / max(taw, 1e-6)) if np.ndim(state["dr"]) else float(state["dr"] / taw),
               eto_3d=float(S.eto[lo:i + 1].sum() / max(1, (i + 1 - lo)) * 24),
               vpd=float(np.mean(_vpd(S.t_air[lo:i + 1], S.rh[lo:i + 1]))),
               rain_3d=float(np.sum(rain_hist[lo:i + 1])) if rain_hist is not None and len(rain_hist) > i else 0.0,
               rain_fc_24h=float(np.nansum(precip_f[:24])) if precip_f is not None else 0.0,
               rain_prob_24h=float(np.nanmax(np.r_[prob_f[:24], 0.0])) if prob_f is not None else 0.0,
               dvs=float(S.dvs[i]), sensitive=bool(eng.crop["sensitive"][0] <= S.dvs[i] <= eng.crop["sensitive"][1]),
               ks=1.0, ece_rel=float((eng.ece_base + float(np.ravel(state["x"])[0])) / eng.crop["ece_thr"]),
               doy=t.timetuple().tm_yday, hour=int(S.local_hour[i]), sand=float(P["sand"]),
               days_since_irr=float(np.ravel(state["h_since_irr"])[0]) / 24.0)
    return ctx


def _sensor_layer(depth_cm, flux, zr):
    """Шар, де стоїть датчик (v1.2): верхній 0–0.3 м, нижній, або «один шар», поки корені ≤ 0.3 м."""
    from .balance import Z_UP
    zu = float(np.mean(flux.get("zu", min(Z_UP, zr))))
    tawu = float(np.mean(flux.get("tawu", 1e9)))
    depth = 20.0 if depth_cm is None else float(depth_cm)
    if zr <= zu + 1e-6:
        return dict(name="single", zu=zu, tawu=tawu)
    return dict(name="upper" if depth <= zu * 100.0 else "lower", zu=zu, tawu=tawu)


def _soil_temp(W):
    """Температура ґрунту з будь-якого доступного шару (Open-Meteo forecast / ERA5-Land мають різні назви)."""
    for k in ("soil_temperature_18cm", "soil_temperature_6cm", "soil_temperature_7_to_28cm",
              "soil_temperature_0_to_7cm"):
        v = W.get(k)
        if isinstance(v, np.ndarray) and np.isfinite(v).any():
            return v
    return None


def _hour_index(times, te):
    """Індекс години, що містить момент te (час у рядах — кінець години)."""
    import bisect
    return min(bisect.bisect_left(times, te), len(times) - 1)


def _vpd(t, rh):
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es * (1 - rh / 100.0)


def _fill_weather_gaps(W):
    for k, v in list(W.items()):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and k != "precipitation_probability":
            if np.isnan(v).any() and (~np.isnan(v)).any():
                idx = np.arange(len(v))
                good = ~np.isnan(v)
                W[k] = np.interp(idx, idx[good], v[good])
    if "surface_pressure" not in W:
        W["surface_pressure"] = None


def _concat_weather(a, b):
    out = {"times": a["times"] + [t for t in b["times"] if t > a["times"][-1]]}
    nb = len(out["times"]) - len(a["times"])
    for k in b:
        if k == "times" or not isinstance(b[k], np.ndarray):
            continue
        av = a.get(k, np.full(len(a["times"]), np.nan))
        out[k] = np.r_[av, b[k][-nb:]] if nb > 0 else av
    return out


def _align_members(ens, times):
    idx = {t: i for i, t in enumerate(ens["times"])}
    cols = [idx.get(t) for t in times]
    if any(c is None for c in cols):
        return None
    m = ens["members"]
    out = np.zeros((m.shape[0], len(times)))
    for j, c in enumerate(cols):
        if c is not None:
            out[:, j] = m[:, c]
    return out


def _initial_depletion(eng, S, W, start, taw0):
    """Стартовий дефіцит: модельна вологість Open-Meteo (слабкий апріорний) + сезонне припущення."""
    month = start.month
    seasonal = 0.15 * taw0 if month in (3, 4, 5, 10, 11) else 0.45 * taw0
    sm = None
    for k in ("soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"):
        v = W.get(k)
        if isinstance(v, np.ndarray) and np.isfinite(v[:24]).any():
            sm = (sm or []) + [float(np.nanmean(v[:24]))]
    if sm:
        th = float(np.mean(sm))
        # шкалуємо відносну вологість моделі NWP на наші FC/WP (у NWP власні ґрунтові константи)
        rel = np.clip((th - 0.10) / (0.38 - 0.10), 0.0, 1.0)
        dr_nwp = (1.0 - rel) * taw0
        dr = 0.5 * dr_nwp + 0.5 * seasonal
        return float(dr), max(0.25 * taw0, 5.0), "Стартова вологість: модель Open-Meteo + сезонне припущення (додайте показ датчика для точності)"
    return float(seasonal), max(0.3 * taw0, 5.0), "Стартова вологість: сезонне припущення (додайте показ датчика для точності)"
