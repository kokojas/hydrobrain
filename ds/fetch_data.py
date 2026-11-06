"""
Завантаження даних для DS-експерименту «прогноз зміни вологості ґрунту на добу вперед».

Джерело: Open-Meteo Historical Weather API: ERA5-Land (~9 км) для ґрунту/температури і ERA5 (~25 км)
для опадів, вітру, радіації та ET0 (у ERA5-Land через Open-Meteo їх немає) — файли <точка>_wx.csv.
  * погода: температура, вологість повітря, точка роси, вітер, радіація, тиск, опади, сніг, ET0 FAO-56;
  * вологість ґрунту в шарах 0–7, 7–28, 28–100 см (м³/м³) — це «правда», яку прогнозуємо.

Точки: 14 сільськогосподарських районів України, 2016–2025 рр.
Додатково: погода для станції FR-Aqui «fraye» мережі ISMN (Франція), де є РЕАЛЬНИЙ датчик вологості
(файл датчика лежить у ds/data/ismn/) — для перевірки методу на справжніх вимірюваннях.

Додатково (--prev-runs): ПРОГНОЗИ погоди, видані за 1 і 2 доби до кожної години (Open-Meteo Previous
Runs API) — файли <точка>_prev.csv. Потрібні, щоб навчати й перевіряти модель на тому, що реально
відомо в момент рішення (прогноз), а не на фактичній погоді ERA5 («ідеальний прогноз» завищує якість).
Архів попередніх прогонів існує не за весь період (орієнтовно з 2024 р., залежить від моделі) —
скрипт сам перевіряє, з якого року є дані, і пропускає роки без них.

Запуск (потрібен лише Python 3.9+, без додаткових пакетів):
    python ds/fetch_data.py                 # реаналіз ERA5-Land/ERA5 (як раніше)
    python ds/fetch_data.py --prev-runs     # + архів прогнозів (можна запускати окремо: --only-prev-runs)
Результат: ds/data/raw/*.csv і архів hydrobrain_ds_data.zip у папці «Завантаження».
Повторний запуск докачує лише відсутнє. Якщо Open-Meteo повідомить про ліміт запитів —
запустіть знову приблизно через годину, скрипт продовжить з того самого місця.
Код виходу: 0 — усе завантажено; 1 — ліміт запитів (запустити пізніше); 2 — частина точок не завантажилась
(список — у locations.json → "failed").

Час у файлах — UTC (GMT). Агрегування до місцевої доби (Europe/Kyiv) — на етапі аналізу.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

URL = "https://archive-api.open-meteo.com/v1/archive"
PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# ERA5-Land в Open-Meteo віддає лише ці змінні; опади, вітер, радіацію, ET0 беремо з ERA5 (fetch_weather).
# (Раніше тут запитувались і погодні змінні — вони приходили порожніми і засмічували CSV.)
HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
          "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
          "soil_moisture_28_to_100cm"]
SOIL = ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"]

# Сільськогосподарські (позаміські) точки в основних аграрних регіонах України
LOCATIONS = [
    ("kherson", 46.75, 33.30), ("mykolaiv", 47.20, 31.80), ("odesa", 46.40, 30.20),
    ("zaporizhzhia", 47.60, 35.60), ("dnipro", 48.30, 34.60), ("kropyvnytskyi", 48.40, 32.60),
    ("poltava", 49.50, 34.20), ("kharkiv", 49.70, 36.60), ("cherkasy", 49.20, 31.70),
    ("vinnytsia", 49.10, 28.70), ("kyiv", 50.20, 30.90), ("zhytomyr", 50.40, 28.30),
    ("ternopil", 49.40, 25.80), ("sumy", 50.80, 34.40),
]
YEARS = (2016, 2025)
# станція ISMN з реальними датчиками (ThetaProbe на 0.05, 0.10, 0.20, 0.30 м), дані 2013-08 … 2022-08
ISMN_SITE = ("fr_aqui_fraye", 44.467, -0.7269, date(2013, 8, 1), date(2022, 8, 9))


class RateLimited(RuntimeError):
    pass


def get_json(params, tries=6, url=URL):
    q = url + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "HydroBrain-DS/1.0 (student project)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:200]
            if e.code == 429:            # ліміт запитів
                if "hour" in body.lower() or "day" in body.lower() or k >= 1:
                    raise RateLimited(body)
                print("    ліміт запитів Open-Meteo, чекаю 65 с …", flush=True)
                time.sleep(65)
                continue
            if e.code == 400:
                raise RuntimeError(f"HTTP 400: {body}")
            time.sleep(5 * (k + 1))
        except Exception as e:           # мережа
            print(f"    мережа: {e}; повтор …", flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("не вдалося отримати дані після кількох спроб")


def fetch_range(lat, lon, d0, d1):
    """Повертає (elevation, model, rows). Спершу ERA5-Land, якщо ґрунтових змінних нема — ERA5."""
    last_err = None
    for model in ("era5_land", "era5", None):
        params = dict(latitude=lat, longitude=lon, start_date=d0.isoformat(), end_date=d1.isoformat(),
                      hourly=",".join(HOURLY), wind_speed_unit="ms", timezone="GMT", timeformat="iso8601")
        if model:
            params["models"] = model
        try:
            data = get_json(params)
        except RuntimeError as e:
            last_err = e
            continue
        h = data.get("hourly", {})
        if not h.get("time") or not all(any(v is not None for v in h.get(s, [])) for s in SOIL):
            last_err = RuntimeError(f"модель {model}: немає вологості ґрунту")
            continue
        rows = []
        for i, t in enumerate(h["time"]):
            rows.append([t] + [h.get(v, [None] * len(h["time"]))[i] for v in HOURLY])
        return data.get("elevation"), model or "best_match", rows
    raise RuntimeError(str(last_err))


def fetch_location(name, lat, lon, d0, d1, out_dir: Path, chunk_years=2):
    path = out_dir / f"{name}.csv"
    if path.exists() and path.stat().st_size > 1000:
        print(f"  {name}: уже є, пропускаю")
        meta = json.loads((out_dir / f"{name}.meta.json").read_text(encoding="utf-8"))
        if "warnings" not in meta:     # мета-файл зі старої версії скрипта — перевіряємо висоту зараз
            elev = meta.get("elevation")
            meta["warnings"] = ([f"висота {elev} м: точка може бути на воді/заплаві/узбережжі — перевірте на карті"]
                                if elev is not None and elev < 5 else [])
            for w_ in meta["warnings"]:
                print(f"    ! {name}: {w_}")
        return meta
    rows, elev, models = [], None, []
    y = d0.year
    while y <= d1.year:
        a = max(d0, date(y, 1, 1))
        b = min(d1, date(y + chunk_years - 1, 12, 31))
        print(f"  {name}: {a} … {b}", flush=True)
        elev, model, r = fetch_range(lat, lon, a, b)
        models.append(dict(start=str(a), end=str(b), model=model))
        rows += r
        y += chunk_years
        time.sleep(1.0)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time"] + HOURLY)
        w.writerows(rows)
    tmp.replace(path)
    used = sorted({m["model"] for m in models})
    warnings = []
    if len(used) > 1:
        # частина періоду не з ERA5-Land → ряди неоднорідні (різна роздільність і ґрунтові константи)
        warnings.append(f"змішані моделі: {used} — див. поле models")
    if elev is not None and elev < 5:
        warnings.append(f"висота {elev} м: точка може бути на воді/заплаві/узбережжі — перевірте на карті")
    for w_ in warnings:
        print(f"    ! {name}: {w_}")
    meta = dict(name=name, lat=lat, lon=lon, elevation=elev, model=used[0] if len(used) == 1 else "mixed",
                models=models, warnings=warnings, start=str(d0), end=str(d1), n_hours=len(rows),
                source="Open-Meteo Historical Weather API (ECMWF ERA5-Land / ERA5)",
                downloaded=datetime.now(timezone.utc).isoformat())
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


WX = ["wind_speed_10m", "shortwave_radiation", "surface_pressure", "precipitation", "snowfall",
      "et0_fao_evapotranspiration"]


def fetch_weather(name, lat, lon, d0, d1, out_dir: Path, chunk_years=2):
    """ERA5-Land в Open-Meteo не містить опадів/вітру/радіації/ET0 → беремо їх з ERA5 (0.25°)."""
    path = out_dir / f"{name}_wx.csv"
    if path.exists() and path.stat().st_size > 1000:
        print(f"  {name}: погода вже є, пропускаю")
        return
    # лише вегетаційний сезон (з запасом на 7-денні ознаки) — удвічі менше запитів до API;
    # кожен рік зберігається окремо, тож після переривання скрипт продовжує з того ж місця
    parts_dir = out_dir / "_parts"
    parts_dir.mkdir(exist_ok=True)
    rows = []
    for y in range(d0.year, d1.year + 1):
        a, b = max(d0, date(y, 3, 20)), min(d1, date(y, 11, 5))
        if a > b:
            continue
        part = parts_dir / f"{name}_{y}.json"
        if part.exists():
            got = json.loads(part.read_text(encoding="utf-8"))
        else:
            print(f"  {name} (погода ERA5): {a} … {b}", flush=True)
            got = None
            for model in ("era5", None):
                params = dict(latitude=lat, longitude=lon, start_date=a.isoformat(), end_date=b.isoformat(),
                              hourly=",".join(WX), wind_speed_unit="ms", timezone="GMT", timeformat="iso8601")
                if model:
                    params["models"] = model
                h = get_json(params).get("hourly", {})
                if h.get("time") and any(v is not None for v in h.get("precipitation", [])):
                    got = h
                    break
            if got is None:
                raise RuntimeError("немає опадів ні в ERA5, ні в best_match")
            part.write_text(json.dumps(got), encoding="utf-8")
            time.sleep(1.0)
        for i, t in enumerate(got["time"]):
            rows.append([t] + [got.get(v, [None] * len(got["time"]))[i] for v in WX])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time"] + WX)
        w.writerows(rows)
    tmp.replace(path)


PREV_VARS = ["precipitation", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation",
             "et0_fao_evapotranspiration"]
PREV_MIN_VARS = ["precipitation", "temperature_2m"]     # якщо повний набір не підтримується
PREV_LEADS = (1, 2)                                     # прогноз, виданий за 1 і 2 доби до цієї години


def _prev_names(vars_):
    return [f"{v}_previous_day{d}" for d in PREV_LEADS for v in vars_]


def fetch_prev_runs(name, lat, lon, d0, d1, out_dir: Path, model=None):
    """
    Архів прогнозів (Open-Meteo Previous Runs API) за вегетаційні сезони.
    Для кожної години: прогноз, виданий за 1 і 2 доби до неї (змінні *_previous_day1/2).
    Роки без даних пропускаються (у API немає архіву за весь період).
    """
    path = out_dir / f"{name}_prev.csv"
    if path.exists() and path.stat().st_size > 1000:
        print(f"  {name}: прогнози вже є, пропускаю")
        with open(path, encoding="utf-8") as f:
            yrs = sorted({ln[:4] for ln in f.readlines()[1:] if ln[:4].isdigit()})
        return dict(name=name, years=[int(y) for y in yrs], file=path.name)
    parts_dir = out_dir / "_parts_prev"
    parts_dir.mkdir(exist_ok=True)
    rows, cols, skipped = [], None, []
    for y in range(d0.year, d1.year + 1):
        a, b = max(d0, date(y, 3, 20)), min(d1, date(y, 11, 5))
        if a > b:
            continue
        part = parts_dir / f"{name}_{y}.json"
        if part.exists():
            got = json.loads(part.read_text(encoding="utf-8"))
        else:
            print(f"  {name} (прогнози, lead 1–2 доби): {a} … {b}", flush=True)
            got = None
            for vars_ in (PREV_VARS, PREV_MIN_VARS):
                params = dict(latitude=lat, longitude=lon, start_date=a.isoformat(), end_date=b.isoformat(),
                              hourly=",".join(_prev_names(vars_)), wind_speed_unit="ms", timezone="GMT",
                              timeformat="iso8601")
                if model:
                    params["models"] = model
                try:
                    h = get_json(params, url=PREV_URL).get("hourly", {})
                except RuntimeError as e:
                    if not str(e).startswith("HTTP 400"):
                        raise                      # мережа/сервер: не позначаємо рік як «порожній»
                    print(f"    {name} {y}: {str(e)[:120]}")   # рік поза архівом або змінна не підтримується
                    continue
                n_ok = sum(v is not None for v in h.get("precipitation_previous_day1", []))
                if h.get("time") and n_ok > 0.5 * len(h["time"]):
                    got = dict(h, _vars=vars_)
                    break
                # відповідь коректна, але значень немає → у архіві немає цього року; менший набір
                # змінних тут не допоможе (він потрібен лише при HTTP 400 «змінна не підтримується»)
                break
            if got is None:
                skipped.append(y)
                part.write_text(json.dumps({"_empty": True}), encoding="utf-8")   # щоб не питати знову
                time.sleep(1.0)
                continue
            part.write_text(json.dumps(got), encoding="utf-8")
            time.sleep(1.0)
        if got.get("_empty"):
            skipped.append(y)
            continue
        names = _prev_names(got["_vars"])
        if cols is None:
            cols = names
        for i, t in enumerate(got["time"]):
            rows.append([t] + [(got.get(c) or [None] * len(got["time"]))[i] if c in got else None for c in cols])
    if not rows:
        print(f"  {name}: архіву прогнозів немає за {d0.year}–{d1.year}")
        return dict(name=name, years=[], file=None)
    if skipped:
        print(f"  {name}: прогнозів немає за роки {skipped} — пропущено")
    have = sorted({int(r[0][:4]) for r in rows})
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time"] + cols)
        w.writerows(rows)
    tmp.replace(path)
    return dict(name=name, years=have, file=path.name)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(here / "data" / "raw"))
    ap.add_argument("--quick", action="store_true", help="лише 3 точки і 2 роки (швидка перевірка)")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--prev-runs", action="store_true", help="також завантажити архів прогнозів (lead 1–2 доби)")
    ap.add_argument("--only-prev-runs", action="store_true", help="лише архів прогнозів (реаналіз уже є)")
    ap.add_argument("--prev-model", default=None, help="модель прогнозу для --prev-runs (типово best_match)")
    ap.add_argument("--sites", default=None,
                    help="лише ці точки, через кому (напр. fr_aqui_fraye); locations.json при цьому не перезаписується")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    locs = LOCATIONS[:3] if a.quick else LOCATIONS
    y0, y1 = (2023, 2024) if a.quick else YEARS
    print(f"Завантаження {len(locs)} точок, {y0}–{y1} → {out}")
    metas, failed, prev = [], [], []
    jobs = [(n, la, lo, date(y0, 1, 1), date(y1, 12, 31)) for n, la, lo in locs] + [ISMN_SITE]
    if a.sites:
        want = {x.strip() for x in a.sites.split(",") if x.strip()}
        jobs = [j for j in jobs if j[0] in want]
        if not jobs:
            print(f"Немає таких точок: {sorted(want)}")
            return 2
    for name, lat, lon, d0, d1 in jobs:
        try:
            if not a.only_prev_runs:
                metas.append(fetch_location(name, lat, lon, d0, d1, out))
                fetch_weather(name, lat, lon, d0, d1, out)
            if a.prev_runs or a.only_prev_runs:
                r = fetch_prev_runs(name, lat, lon, d0, d1, out, model=a.prev_model)
                if r:
                    prev.append(r)
        except RateLimited:
            print("\nOpen-Meteo: вичерпано ліміт безкоштовних запитів на цю годину/добу.\n"
                  "Усе вже завантажене збережено. Запустіть скрипт ще раз приблизно через годину —\n"
                  "він продовжить з того самого місця.")
            return 1
        except Exception as e:
            # не «ковтаємо» помилку мовчки: точка потрапляє у список failed і впливає на код виходу
            print(f"  ! {name}: {e}")
            failed.append(dict(name=name, error=str(e)[:300]))
    if not a.only_prev_runs and not a.sites:
        summary = dict(locations=metas, failed=failed,
                       warnings={m["name"]: m.get("warnings") for m in metas if m.get("warnings")})
        (out / "locations.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if metas:
        print(f"Реаналіз: готово, точок: {len(metas)}.")
    if prev:
        (out / "prev_runs.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
        ok = [p_ for p_ in prev if p_["years"]]
        print(f"Архів прогнозів: є для {len(ok)} з {len(prev)} точок.")
        for p_ in prev:
            print(f"  {p_['name']:15s} роки: {p_['years'] or '—'}")
    if failed:
        print(f"НЕ завантажено: {[f['name'] for f in failed]}")
    if not a.no_zip:
        dl = Path.home() / "Downloads"
        target = (dl if dl.exists() else here) / "hydrobrain_ds_data.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.iterdir()):
                if p.is_file():
                    z.write(p, arcname=f"raw/{p.name}")
        print(f"Архів для передачі: {target}  ({target.stat().st_size / 1e6:.1f} МБ)")
    if failed:
        print("УВАГА: частина точок не завантажилась — запустіть скрипт ще раз (він докачає лише відсутнє).")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
