"""
Каталог параметрів (групи 0-7 зі схеми проєкту + датчики) і «розв'язувач» значень
з урахуванням режиму користувача:

  beginner (новачок) — вводить лише кілька полів (де поле, культура, дата сівби, спосіб поливу,
                       за бажанням тип ґрунту і датчик); решта береться з карт/API/довідників;
  expert   (розширений) — може задати будь-який параметр; незаданий — так само з API/довідників.

Для кожного параметра зберігається ПОХОДЖЕННЯ (користувач / API / довідник / за замовчуванням /
розраховано) — це потрібно для пояснень і для оцінки невизначеності σ_geo.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone

from .databases import (CROPS, DRAINAGE_CLASSES, FERTILIZERS, GEOLOGY, IRRIGATION_METHODS,
                        SOILS_UA)


@dataclass
class ParamSpec:
    key: str
    symbol: str
    group: int
    name_uk: str
    unit: str = ""
    kind: str = "float"          # float | int | select | date | bool | text | list
    default: object = None
    options: dict = field(default_factory=dict)
    beginner: bool = False
    effect_uk: str = ""
    min: float | None = None
    max: float | None = None
    auto: str = ""               # звідки береться автоматично


def _opts(db):
    return {k: v["name_uk"] for k, v in db.items()}


GROUP_NAMES = {
    0: "Геологія та локація", 1: "Ґрунтові фактори", 2: "Кліматичні / атмосферні",
    3: "Біологічні (рослина)", 4: "Менеджмент / технології / добрива", 5: "Поле / мікроклімат",
    6: "Просторово-часові", 7: "Обмеження / невизначеності", 8: "Датчики",
}

CATALOG = [
    # ---- 0. Геологія та локація
    ParamSpec("lat", "Loc", 0, "Широта поля", "°", default=None, beginner=True, min=-90, max=90,
              effect_uk="клімат, радіація, прогноз"),
    ParamSpec("lon", "Loc", 0, "Довгота поля", "°", default=None, beginner=True, min=-180, max=180,
              effect_uk="клімат, радіація, прогноз"),
    ParamSpec("alt", "Alt", 0, "Висота над рівнем моря", "м", default=150.0, min=-30, max=2100,
              effect_uk="↓ тиск Pa → γ; ↑ амплітуда T", auto="Open-Meteo Elevation (Copernicus DEM)"),
    ParamSpec("geo", "Geo", 0, "Тип корінної породи", kind="select", default="loess", options=_opts(GEOLOGY),
              effect_uk="базові Ds_max, дренаж, ECe, pH", auto="BGR IGME5000 / Macrostrat"),
    ParamSpec("drainage_class", "Dr", 0, "Дренажний клас", kind="select", default=None,
              options=_opts(DRAINAGE_CLASSES), effect_uk="DPt (швидкість дренажу), ROt",
              auto="з геології та текстури"),
    ParamSpec("ds_max", "Ds_max", 0, "Макс. глибина ґрунту до породи", "м", default=None, min=0.2, max=5,
              effect_uk="обмежує корені", auto="з геології"),
    # ---- 1. Ґрунтові
    ParamSpec("soil_type_ua", "Tx", 1, "Тип ґрунту (довідник України)", kind="select", default="auto",
              options={"auto": "Визначити автоматично (SoilGrids)", **_opts(SOILS_UA)}, beginner=True,
              effect_uk="задає текстуру, гумус, щільність, pH, EC"),
    ParamSpec("sand", "Tx", 1, "Вміст піску", "%", min=0, max=100, effect_uk="DP / дренаж, AWC",
              auto="SoilGrids / довідник"),
    ParamSpec("clay", "Tx", 1, "Вміст глини", "%", min=0, max=80, effect_uk="AWC, стік",
              auto="SoilGrids / довідник"),
    ParamSpec("om", "OM", 1, "Органічна речовина (гумус)", "%", min=0, max=12, effect_uk="↑ AWC",
              auto="SoilGrids (SOC×1.724) / довідник"),
    ParamSpec("bd", "BD", 1, "Щільність ґрунту", "г/см³", min=0.9, max=1.9,
              effect_uk="↓ пористість → ↓ інфільтрація", auto="SoilGrids / довідник"),
    ParamSpec("structure", "Agg", 1, "Структура (агрегатність)", kind="select", default=None,
              options={"good": "Добра (зерниста)", "medium": "Середня", "poor": "Погана (брила/кірка)"},
              effect_uk="водопроникність, стік", auto="довідник"),
    ParamSpec("ds", "Ds", 1, "Ефективна глибина ґрунту", "м", min=0.2, max=4, effect_uk="резервуар вологи",
              auto="довідник / геологія"),
    ParamSpec("theta_fc", "AWC", 1, "Вологість польової вологоємності θFC", "м³/м³", min=0.05, max=0.6,
              effect_uk="буфер вологи Wt", auto="педотрансфер Saxton & Rawls"),
    ParamSpec("theta_wp", "AWC", 1, "Вологість в'янення θWP", "м³/м³", min=0.01, max=0.4,
              effect_uk="буфер вологи Wt", auto="педотрансфер Saxton & Rawls"),
    ParamSpec("ksat_mm_h", "Ksat", 1, "Коефіцієнт фільтрації Ksat", "мм/год", min=0.1, max=500,
              effect_uk="дренаж, група стоку", auto="педотрансфер"),
    ParamSpec("ece_soil", "ECt", 1, "Засоленість ґрунту ECe (базова)", "дСм/м", min=0, max=20,
              effect_uk="солевий стрес Kstress", auto="довідник"),
    ParamSpec("ph", "pH", 1, "pH ґрунту", "", min=3.5, max=10, effect_uk="доступність води/живлення",
              auto="SoilGrids / довідник"),
    ParamSpec("gw_depth", "GW", 1, "Глибина ґрунтових вод", "м", min=0.3, max=30,
              effect_uk="капілярне підживлення CRt", auto="довідник (якщо відомо)"),
    ParamSpec("hydrophobicity", "Hf", 1, "Гідрофобність ґрунту", "0..1", default=0.0, min=0, max=1,
              effect_uk="тимчасове відштовхування води → стік"),
    ParamSpec("cn_lambda", "λ", 1, "Частка початкових втрат у методі CN (Ia = λ·S)", kind="select", default=0.2,
              options={0.2: "0.2 — стандарт NEH-630", 0.05: "0.05 — Hawkins et al. (2002)"},
              effect_uk="стік від злив; 0.05 дає більше стоку від невеликих дощів"),
    ParamSpec("surface_storage_mm", "S_d", 4, "Поверхнева ємність поля між циклами поливу", "мм", default=None,
              min=0.5, max=15.0, effect_uk="скільки води втримує поверхня (западини, мульча) під час циклу; "
              "типово 3 мм; визначає тривалість циклів, коли інтенсивність дощування > Ksat"),
    ParamSpec("application_rate_mm_h", "I_rate", 4, "Інтенсивність дощування", "мм/год", default=None, min=0.5,
              max=200, effect_uk="якщо більше Ksat — частина поливу стікає", auto="витрата системи / площа"),
    ParamSpec("hsg", "HSG", 1, "Гідрологічна група ґрунту", kind="select", default=None,
              options={"A": "A", "B": "B", "C": "C", "D": "D"}, auto="з Ksat"),
    # ---- 2. Кліматичні (самі метеодані — з API; тут налаштування)
    ParamSpec("albedo", "α", 2, "Альбедо поверхні", "", default=0.23, min=0.05, max=0.6,
              effect_uk="↓ ETo при світлій поверхні"),
    ParamSpec("co2_ppm", "CO2", 2, "Концентрація CO2 (поправка опору продихів)", "ppm", default=None,
              min=280, max=1000, effect_uk="↓ транспірація (вимкнено, якщо порожньо)"),
    ParamSpec("weather_model", "—", 2, "Модель прогнозу Open-Meteo", kind="select", default="best_match",
              options={"best_match": "Автовибір (рекомендовано)", "ecmwf_ifs025": "ECMWF IFS 0.25°",
                       "icon_seamless": "DWD ICON (EU+global)", "gfs_seamless": "NOAA GFS"}),
    ParamSpec("use_ensemble", "σ_fc", 2, "Брати ансамбль для невизначеності опадів", kind="bool", default=True),
    ParamSpec("pstress_calibration", "Pcal", 2, "Калібрувати ймовірність стресу", kind="bool", default=False,
              effect_uk=("ймовірність стресу коригується кривою, підібраною на бектесті (1.11): без неї ймовірності "
                         "систематично не збігались із частотою подій")),
    ParamSpec("forecast_core", "ML", 2, "Ядро прогнозу природної зміни вологості", kind="select", default="auto",
              options={"auto": "Автоматично (рекомендовано): ML в Україні, фізика поза нею",
                       "ml": "ML завжди (навчена на ERA5-Land України) + фізика для поливу",
                       "physics": "Лише фізична модель FAO-56"},
              effect_uk=("ML точніша на всіх перевірках: ERA5-Land України зі справжнім прогнозом погоди — MAE 0.10 "
                         "проти 0.19 % об./добу у фізики; реальні датчики 0–55 см — 0.28 проти 1.00. Фізику "
                         "обирайте для поля поза Україною (поза умовами навчання ML) або для порівняння. Полив, "
                         "поріг культури і норму в усіх варіантах рахує фізика.")),
    # ---- 3. Біологічні
    ParamSpec("crop", "Sp", 3, "Культура", kind="select", default="maize", options=_opts(CROPS), beginner=True,
              effect_uk="базовий профіль Kc, корені, p"),
    ParamSpec("sowing_date", "GS", 3, "Дата сівби / відновлення вегетації", kind="date", default=None,
              beginner=True, effect_uk="стадія росту → форма Kc(t)"),
    ParamSpec("dev_mode", "GS", 3, "Спосіб рахунку фаз", kind="select", default="thermal",
              options={"thermal": "За температурою (бета-модель Yin)", "calendar": "За календарем FAO-56"}),
    ParamSpec("kc_mid_override", "Kc", 3, "Власний Kcb середини сезону", "", min=0.3, max=1.4,
              effect_uk="ETc = Kc×ETo"),
    ParamSpec("zr_max_override", "Rd", 3, "Макс. глибина коренів", "м", min=0.1, max=3,
              effect_uk="доступ до глибоких шарів"),
    ParamSpec("lai_measured", "LAI", 3, "Індекс листкової поверхні (виміряний)", "м²/м²", min=0, max=10,
              effect_uk="↑ транспірація"),
    ParamSpec("plant_health", "PH", 3, "Здоров'я рослин", "0..1", default=1.0, min=0.2, max=1.0,
              effect_uk="хвороби/шкідники → ↓ транспірація"),
    ParamSpec("plant_density", "PD", 3, "Густота посіву", "росл./м²", min=0.1, max=1000,
              effect_uk="швидке вичерпання Wt при надмірній"),
    ParamSpec("plant_density_opt", "PD", 3, "Рекомендована густота", "росл./м²", min=0.1, max=1000),
    ParamSpec("gs", "gs", 3, "Провідність продихів (виміряна)", "ммоль/м²/с", min=0, max=1500,
              effect_uk="пряма зміна транспірації"),
    ParamSpec("gs_ref", "gs", 3, "Еталонна провідність продихів", "ммоль/м²/с", default=400.0),
    # ---- 4. Менеджмент
    ParamSpec("irrigation_method", "Im", 4, "Спосіб поливу", kind="select", default="drip",
              options=_opts(IRRIGATION_METHODS), beginner=True, effect_uk="ефективність, змочена частка"),
    ParamSpec("ei_mult", "Ei", 4, "Поправка ефективності системи", "×", default=1.0, min=0.5, max=1.15,
              effect_uk="0.9 / 0.7 … (стан системи)"),
    ParamSpec("water_temp", "Tw", 4, "Температура поливної води", "°C", min=0, max=40, effect_uk="K_stress"),
    ParamSpec("ec_iw", "Sw", 4, "Якість води: ECiw", "дСм/м", default=0.3, min=0, max=10,
              effect_uk="K_stress, накопичення солей"),
    ParamSpec("fertilizers", "Fd/Ft", 4, "Внесення добрив (список подій)", kind="list", default=[],
              effect_uk="↑ ECt (сольовий індекс)"),
    ParamSpec("allow_leaching", "Fl", 4, "Додавати норму промивання солей", kind="bool", default=True,
              effect_uk="спад ECt"),
    ParamSpec("prev_crop", "Hist", 4, "Попередник (історія сівозміни)", kind="select", default="unknown",
              options={"unknown": "Невідомо", "fallow": "Пар", "sunflower": "Соняшник", "maize": "Кукурудза",
                       "winter_wheat": "Озима пшениця", "sugar_beet": "Буряк", "lucerne": "Люцерна"},
              effect_uk="залишкова волога підґрунтя"),
    ParamSpec("automation", "Auto", 4, "Автоматизація поливу", kind="select", default="manual",
              options={"manual": "Вручну", "timer": "Таймер", "auto": "Автоматика з лічильником"},
              effect_uk="похибка фактичної норми It"),
    ParamSpec("mulch_type", "Mu", 4, "Мульчування", kind="select", default="none",
              options={"none": "Немає", "organic": "Органічна (солома, рештки)", "plastic": "Плівка"},
              beginner=True, effect_uk="↓ випаровування (30-80%)"),
    ParamSpec("mulch_cover", "Mu", 4, "Частка поверхні під мульчею", "0..1", default=0.5, min=0, max=1),
    ParamSpec("area_m2", "—", 4, "Площа поля / ділянки", "м²", default=None, beginner=True, min=1),
    ParamSpec("flow_rate_m3_h", "—", 4, "Витрата системи поливу", "м³/год", default=None, beginner=True, min=0.001),
    ParamSpec("max_single_mm", "Sch", 4, "Макс. разова норма", "мм", default=None, min=1, max=150),
    ParamSpec("preferred_hours", "ToD", 4, "Дозволені години поливу (місцевий час)", kind="list",
              default=list(range(4, 10)) + list(range(18, 23))),
    # ---- 5. Поле / мікроклімат
    ParamSpec("stones", "Soil_f", 5, "Кам'янистість (об'ємна частка)", "0..1", default=None, min=0, max=0.8,
              effect_uk="↓ AWC локально", auto="SoilGrids cfvo"),
    ParamSpec("shade", "Sh", 5, "Затінення прямого сонця", "0..1", default=0.0, min=0, max=0.95,
              effect_uk="↓ Rs → ↓ ETo"),
    ParamSpec("vent", "Vent", 5, "Вентиляція / захищеність від вітру", kind="select", default="normal",
              options={"sheltered": "Захищене (лісосмуга, двір)", "normal": "Звичайне поле",
                       "exposed": "Відкрите, вітряне"}, effect_uk="↑ локальний ETo"),
    ParamSpec("slope_deg", "Sl", 5, "Ухил", "°", default=None, min=0, max=45, effect_uk="↑ радіація/стік",
              auto="DEM (метод Горна)"),
    ParamSpec("aspect_deg", "Exp", 5, "Експозиція схилу (азимут)", "°", default=None, min=0, max=360,
              effect_uk="південні схили отримують більше радіації", auto="DEM"),
    # ---- 6. Просторово-часові
    ParamSpec("horizon_h", "Hz", 6, "Горизонт планування", "год", default=72, kind="int", min=24, max=240,
              effect_uk="глибина прогнозу"),
    ParamSpec("tz_offset_h", "ToD", 6, "Зсув місцевого часу від UTC", "год", default=None, min=-12, max=14,
              auto="Europe/Kyiv (літній/зимовий час)"),
    # ---- 7. Невизначеності
    ParamSpec("sigma_sens", "σ_sens", 7, "Випадкова похибка датчика вологості", "м³/м³", default=0.01, min=0.002, max=0.1),
    ParamSpec("sigma_spat", "σ_spat", 7, "Просторова варіабельність поля", "м³/м³", default=0.015, min=0, max=0.1),
    ParamSpec("sigma_meas", "σ_meas", 7, "Похибка вхідних метеоданих (відносна)", "", default=0.08, min=0, max=0.5),
    ParamSpec("sigma_geo", "σ_geo", 7, "Невизначеність запасу вологи TAW (відносна)", "", default=None,
              min=0.02, max=0.6, auto="з походження ґрунтових даних"),
    ParamSpec("mc_samples", "N", 7, "Кількість сценаріїв Монте-Карло", "", default=200, kind="int", min=20, max=2000),
    # ---- 8. Датчики
    ParamSpec("sensor_depth_cm", "θ(z)", 8, "Глибина встановлення датчика", "см", default=20, beginner=True,
              min=2, max=150),
    ParamSpec("sensor_calibrated", "b", 8, "Датчик відкалібровано під цей ґрунт (ваговим методом)", kind="bool",
              default=False, effect_uk="зменшує невизначеність зміщення датчика з ±2% до ±0.8%"),
    ParamSpec("sensor_calib", "b", 8, "Калібрування датчика θ = a + b·показ (список [a, b])", kind="list",
              default=None, effect_uk="перерахунок показу в справжню вологість за ваговими пробами"),
    ParamSpec("sensor_offset", "b", 8, "Відома поправка калібрування датчика", "м³/м³", default=0.0,
              min=-0.2, max=0.2),
]

SPEC = {p.key: p for p in CATALOG}


def validate_mode(mode):
    if mode not in ("beginner", "expert"):
        raise ValueError("Режим має бути beginner або expert")


def _validate_value(p, value):
    """Normalize user data before any network calls or state writes."""
    label = p.name_uk
    if p.kind in ("float", "int"):
        if isinstance(value, bool):
            raise ValueError(f"{label}: потрібне число")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{label}: потрібне число") from None
        if not math.isfinite(number):
            raise ValueError(f"{label}: число має бути скінченним")
        if p.kind == "int" and not number.is_integer():
            raise ValueError(f"{label}: потрібне ціле число")
        if (p.min is not None and number < p.min) or (p.max is not None and number > p.max):
            raise ValueError(f"{label}: значення поза межами {p.min} … {p.max}")
        return int(number) if p.kind == "int" else number
    if p.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{label}: потрібне true або false")
    elif p.kind == "select":
        # HTML select sends numeric option keys as strings (cn_lambda).
        matches = [k for k in p.options if str(k) == str(value)]
        if not matches:
            raise ValueError(f"{label}: невідомий варіант {value!r}")
        return matches[0]
    elif p.kind == "date":
        try:
            return date.fromisoformat(value).isoformat()
        except (TypeError, ValueError):
            raise ValueError(f"{label}: потрібна дата РРРР-ММ-ДД") from None
    elif p.kind == "list":
        if not isinstance(value, list):
            raise ValueError(f"{label}: потрібен список")
        if p.key == "sensor_calib":
            if len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                      or not math.isfinite(v) for v in value) or value[1] <= 0:
                raise ValueError("sensor_calib: потрібні два скінченні числа [a, b], b > 0")
        elif p.key == "preferred_hours":
            if any(type(v) is not int or not 0 <= v <= 23 for v in value):
                raise ValueError("preferred_hours: потрібні цілі години 0 … 23")
        elif p.key == "fertilizers":
            for event in value:
                if not isinstance(event, dict):
                    raise ValueError("fertilizers: кожна подія має бути об’єктом")
                try:
                    datetime.fromisoformat(event["time"].replace("Z", "+00:00"))
                    dose = float(event["dose_kg_ha"])
                    valid = math.isfinite(dose) and dose >= 0 and event.get("type", "npk_complex") in FERTILIZERS
                except (KeyError, TypeError, ValueError, AttributeError):
                    valid = False
                if not valid:
                    raise ValueError("fertilizers: перевірте time, dose_kg_ha і type")
    return value


def input_schema(mode="beginner"):
    """Опис полів для автоматичної побудови форми в інтерфейсі."""
    validate_mode(mode)
    rows = []
    for p in CATALOG:
        if mode == "beginner" and not p.beginner:
            continue
        d = asdict(p)
        d["group_name"] = GROUP_NAMES[p.group]
        rows.append(d)
    return rows


# ---------------------------------------------------------------------------
class Resolver:
    """Збирає повний набір параметрів: користувач → API → довідники → за замовчуванням → розраховано."""

    SIGMA_BY_SOURCE = {"user": 0.08, "api": 0.15, "preset": 0.20, "default": 0.30}

    def __init__(self, sources=None):
        self.sources = sources

    def resolve(self, user: dict, mode="beginner"):
        validate_mode(mode)
        if not isinstance(user, dict):
            raise ValueError("Параметри поля мають бути об’єктом")
        warnings, prov, P = [], {}, {}
        allowed = {p.key for p in CATALOG if (mode == "expert" or p.beginner)}
        u = {}
        for k, v in (user or {}).items():
            if v is None or v == "":
                continue
            if k not in SPEC:
                warnings.append(f"Невідомий параметр «{k}» — проігноровано")
                continue
            if k not in allowed:
                warnings.append(f"«{SPEC[k].name_uk}» доступний лише в розширеному режимі — взято автоматичне значення")
                continue
            u[k] = _validate_value(SPEC[k], v)

        def put(k, v, src, note=""):
            P[k] = v
            prov[k] = dict(value=v, source=src, note=note, symbol=SPEC[k].symbol if k in SPEC else "",
                           name_uk=SPEC[k].name_uk if k in SPEC else k)

        # --- локація
        if "lat" not in u or "lon" not in u:
            raise ValueError("Потрібні координати поля (lat, lon)")
        put("lat", float(u["lat"]), "user")
        put("lon", float(u["lon"]), "user")
        if not (44 <= P["lat"] <= 53 and 22 <= P["lon"] <= 41):
            warnings.append("Координати поза межами України — довідники розраховані на українські умови")

        # --- рельєф
        terrain = None
        if self.sources is not None:
            try:
                terrain = self.sources.terrain(P["lat"], P["lon"])
            except Exception as e:
                warnings.append(f"Рельєф (DEM) недоступний: {e}")
        for k, dflt in (("alt", 150.0), ("slope_deg", 0.0), ("aspect_deg", 180.0)):
            if k in u:
                put(k, float(u[k]), "user")
            elif terrain:
                put(k, terrain[k], "api", "Open-Meteo Elevation / Copernicus DEM")
            else:
                put(k, dflt, "default")

        # --- геологія
        geo_info = None
        if "geo" in u:
            put("geo", u["geo"], "user")
        else:
            if self.sources is not None:
                try:
                    geo_info = self.sources.geology(P["lat"], P["lon"])
                except Exception as e:
                    warnings.append(f"Геологія: {e}")
            if geo_info and geo_info["geo"] != "unknown":
                put("geo", geo_info["geo"], "api", f'{geo_info["source"]}: {geo_info["description"][:120]}')
            else:
                put("geo", "loess", "default", "лес — найпоширеніша материнська порода в Україні")
        g = GEOLOGY.get(P["geo"], GEOLOGY["unknown"])

        # --- ґрунт
        preset_key = u.get("soil_type_ua", "auto")
        soil_src = "default"
        base = dict(SOILS_UA["chornozem_typical"])
        if preset_key != "auto" and preset_key in SOILS_UA:
            base = dict(SOILS_UA[preset_key])
            soil_src = "preset"
            put("soil_type_ua", preset_key, "user")
        sg = None
        if self.sources is not None and (soil_src != "preset" or mode == "expert"):
            try:
                sg = self.sources.soilgrids(P["lat"], P["lon"])
            except Exception as e:
                warnings.append(f"SoilGrids недоступний: {e}")
        if soil_src != "preset":
            if sg:
                if sg.get("note"):
                    warnings.append("SoilGrids: " + sg["note"])
                soil_src = "api"
                put("soil_type_ua", "auto", "api", sg["source"])
            else:
                put("soil_type_ua", "chornozem_typical", "default",
                    "API недоступні — взято чорнозем типовий; оберіть тип ґрунту вручну для точності")
                warnings.append("Тип ґрунту не визначено автоматично — використано чорнозем типовий")
        for k in ("sand", "clay", "om", "bd", "ph"):
            if k in u:
                put(k, float(u[k]), "user")
            elif soil_src == "api" and sg and k in sg:
                put(k, round(float(sg[k]), 2), "api", sg["source"])
            else:
                put(k, base[k], "preset" if soil_src == "preset" else "default", base["name_uk"])
        if sg and soil_src == "preset" and abs(sg["clay"] - base["clay"]) > 15:
            warnings.append(f"Вибраний тип ґрунту має глини {base['clay']}%, а SoilGrids показує "
                            f"{sg['clay']:.0f}% — перевірте тип ґрунту")
        for k, bk in (("ece_soil", "ece"), ("structure", "structure"), ("ds", "depth")):
            if k in u:
                put(k, u[k] if k == "structure" else float(u[k]), "user")
            else:
                put(k, base[bk], "preset" if soil_src == "preset" else "default", base["name_uk"])
        if "stones" in u:
            put("stones", float(u["stones"]), "user")
        elif sg and "stones" in sg:
            put("stones", round(min(0.6, sg["stones"]), 3), "api", sg["source"])
        else:
            put("stones", 0.0, "default")
        if "gw_depth" in u:
            put("gw_depth", float(u["gw_depth"]), "user")
        else:
            put("gw_depth", base.get("gw"), "preset" if soil_src == "preset" else "default",
                "типова для цього ґрунту (None = глибоко)")
        for k in ("theta_fc", "theta_wp", "ksat_mm_h", "hsg"):
            if k in u:
                put(k, u[k], "user")
            elif k in ("theta_fc", "theta_wp") and soil_src in ("preset", "default") and base.get(k[6:]) is not None \
                    and not any(t in u for t in ("sand", "clay", "om", "bd")):
                # виміряні константи з довідника мають пріоритет над педотрансфером — але лише тоді, коли
                # користувач не задав власну текстуру ґрунту (1.8: інакше пісок отримував θFC чорнозему)
                put(k, base[k[6:]], "preset", base.get("src", ""))
            else:
                put(k, None, "derived", "Saxton & Rawls 2006 / NEH-630")
        # геологічні похідні
        put("geo_ece_add", g["ece_add"], "derived", g["name_uk"])
        if "ds_max" in u:
            put("ds_max", float(u["ds_max"]), "user")
        else:
            put("ds_max", g["ds_max"], "derived", g["name_uk"])
        if "drainage_class" in u:
            put("drainage_class", u["drainage_class"], "user")
        else:
            dc = g["drainage"]
            if P["sand"] > 70:
                dc = "excessive"
            elif P["clay"] > 40 and dc == "good":
                dc = "moderate"
            put("drainage_class", dc, "derived", "з геології та текстури")
        put("ph", round(P["ph"] + (0 if "ph" in u else g["ph_shift"] * 0.3), 2), prov["ph"]["source"],
            prov["ph"]["note"])

        # --- культура, менеджмент, мікроклімат і решта — значення користувача або за замовчуванням
        for p in CATALOG:
            if p.key in P:
                continue
            if p.key in u:
                put(p.key, u[p.key], "user")
            else:
                put(p.key, p.default, "default")
        if P["crop"] not in CROPS:
            raise ValueError(f"Невідома культура {P['crop']}")
        if P["irrigation_method"] not in IRRIGATION_METHODS:
            raise ValueError(f"Невідомий спосіб поливу {P['irrigation_method']}")
        if not P.get("sowing_date"):
            today = datetime.now(timezone.utc).date()
            put("sowing_date", date(today.year, 4, 25).isoformat(), "default",
                "дата сівби не вказана — взято 25 квітня поточного року")
            warnings.append("Не вказано дату сівби — стадія росту оцінена приблизно")
        if P.get("structure") is None:
            P["structure"] = "medium"

        if P["sand"] + P["clay"] > 100:
            raise ValueError("Сума часток піску та глини не може перевищувати 100%")
        # Check the effective constants, including a derived counterpart.
        from .soil import saxton_rawls
        hyd = saxton_rawls(P["sand"], P["clay"], P["om"], P["bd"])
        fc = P["theta_fc"] if P["theta_fc"] is not None else hyd["theta_fc"]
        wp = P["theta_wp"] if P["theta_wp"] is not None else hyd["theta_wp"]
        if fc <= wp:
            raise ValueError("Вологість θFC має бути більшою за θWP")

        # σ_geo — з походження ґрунтових даних і геології
        if P.get("sigma_geo") is None:
            put("sigma_geo", max(self.SIGMA_BY_SOURCE.get(prov["sand"]["source"], 0.3), g["taw_sigma"] * 0.8),
                "derived", "з джерела ґрунтових даних")
        return P, prov, warnings
