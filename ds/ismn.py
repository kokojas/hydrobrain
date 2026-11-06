"""
Читання даних реальних датчиків вологості ґрунту з ISMN (International Soil Moisture Network).

Як отримати файли:
  1. Зареєструватися на https://ismn.earth (безкоштовно) → Data Viewer.
  2. Вибрати мережу/станцію (наприклад, FR_Aqui → Fraye), змінну soil_moisture, період.
  3. Формат завантаження — «Header + values» (файли *.stm). Розпакувати архів у ds/data/ismn/.
     Структура всередині архіву не важлива: функція find_files шукає *.stm рекурсивно.

Формат *.stm (Header + values):
  перший рядок — заголовок:  <мережа> <мережа> <станція> <lat> <lon> <висота> <глибина_від> <глибина_до> <датчик>
  далі рядки:               2016/05/01 13:00 0.2345 G M
                             дата час(UTC) значення(м³/м³) прапорець_якості_ISMN прапорець_постачальника
  Прапорець «G» = good. Інші (D01…D10, C01…C03 тощо) — підозрілі значення, типово їх відкидаємо.

Альтернатива: офіційний пакет `pip install ismn` читає архів напряму. Цей модуль — без залежностей,
крім pandas, і дає рівно те, що потрібно для порівняння з ERA5-Land / моделлю.

Використання:
    from ds.ismn import load_station
    df = load_station("ds/data/ismn", station="Fraye", depth_max=0.10)   # погодинно, UTC
    daily = to_daily(df, tz="Europe/Paris")
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_DATA_RE = re.compile(r"^(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2})\s+(-?[\d.]+(?:[eE][-+]?\d+)?)\s*(\S*)\s*(\S*)")


def read_stm(path) -> pd.DataFrame:
    """Один файл *.stm → DataFrame[time (UTC), sm, flag, flag_provider] + df.attrs із заголовка."""
    path = Path(path)
    rows, header = [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _DATA_RE.match(line.strip())
            if m:
                d, t, v, fl, fp = m.groups()
                rows.append((f"{d} {t}", float(v), fl or "", fp or ""))
            elif header is None and line.strip():
                header = line.split()
    if not rows:
        raise ValueError(f"{path.name}: не знайдено жодного рядка даних (це точно формат Header+values?)")
    df = pd.DataFrame(rows, columns=["time", "sm", "flag", "flag_provider"])
    df["time"] = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M", utc=True)
    attrs = dict(file=path.name)
    if header and len(header) >= 8:
        try:
            attrs.update(network=header[1], station=header[2], lat=float(header[3]), lon=float(header[4]),
                         elevation=float(header[5]), depth_from=float(header[6]), depth_to=float(header[7]),
                         sensor=" ".join(header[8:]) or None)
        except ValueError:
            attrs["header_raw"] = " ".join(header)
    df.attrs = attrs
    return df


def find_files(root, station: str | None = None, variable: str = "sm") -> list[Path]:
    """Усі *.stm під root; фільтр за назвою станції і змінною (у назвах файлів ISMN є «_sm_»)."""
    out = []
    for p in sorted(Path(root).rglob("*.stm")):
        name = p.name.lower()
        if f"_{variable}_" not in name:
            continue
        # точний збіг назви станції (у назві файлу вона стоїть між назвою мережі і «_<змінна>_»):
        # підрядок не годиться — «hillan» інакше збігся б з «hillan2»
        if station and f"_{station.lower()}_{variable}_" not in name and p.parent.name.lower() != station.lower():
            continue
        out.append(p)
    return out


def load_station(root, station: str | None = None, depth_max: float = 0.10, good_only: bool = True,
                 valid_range=(0.0, 0.7)) -> pd.DataFrame:
    """
    Погодинний ряд вологості станції: усі датчики з глибиною до depth_max (м) усереднюються.
    Відкидаються значення з прапорцем ≠ G (якщо good_only) і поза фізичним діапазоном.
    Повертає DataFrame з індексом time (UTC) і колонками sm, n_sensors.
    """
    files = find_files(root, station)
    if not files:
        raise FileNotFoundError(f"У {root} немає файлів *.stm для станції {station!r}. "
                                f"Завантажте їх з ismn.earth у форматі «Header + values».")
    parts, used = [], []
    for p in files:
        df = read_stm(p)
        dt = df.attrs.get("depth_to")
        if dt is not None and dt > depth_max:
            continue
        if good_only:
            df = df[df["flag"].str.startswith("G")]
        df = df[(df["sm"] >= valid_range[0]) & (df["sm"] <= valid_range[1])]
        if df.empty:
            continue
        parts.append(df.set_index("time")["sm"].rename(p.stem))
        used.append(dict(df.attrs))
    if not parts:
        raise ValueError(f"Немає придатних даних до глибини {depth_max} м (перевірте depth_max і прапорці).")
    wide = pd.concat(parts, axis=1).sort_index()
    wide = wide[~wide.index.duplicated()]
    out = pd.DataFrame({"sm": wide.mean(axis=1), "n_sensors": wide.notna().sum(axis=1)})
    out.attrs["sensors"] = used
    return out


def to_daily(df: pd.DataFrame, tz: str = "UTC", min_hours: int = 18) -> pd.DataFrame:
    """Добові середні за місцевою добою; доби з < min_hours вимірів відкидаються (неповні)."""
    s = df["sm"].tz_convert(tz) if df.index.tz is not None else df["sm"]
    g = s.resample("D")
    d = pd.DataFrame({"sm": g.mean(), "n_hours": g.count()})
    return d[d["n_hours"] >= min_hours]
