"""
Зовнішні джерела даних (усі безкоштовні, без ключа для некомерційного використання):

  ПОГОДА
  * Open-Meteo Forecast API   https://api.open-meteo.com/v1/forecast
      моделі ECMWF IFS, DWD ICON-EU/ICON, … ; до 16 діб прогнозу, до 92 діб минулого (past_days);
      змінні: T, RH, вітер, радіація, тиск, опади, ймовірність опадів, ET0 FAO-56,
      температура і вологість ґрунту по шарах.
  * Open-Meteo Ensemble API   https://ensemble-api.open-meteo.com/v1/ensemble
      десятки членів ансамблю → реальна невизначеність опадів (σ_fc).
  * Open-Meteo Historical API https://archive-api.open-meteo.com/v1/archive  (ERA5, для довгої історії)
  * NASA POWER (AG)           https://power.larc.nasa.gov/api/temporal/daily/point  (резерв/кліматологія)

  РЕЛЬЄФ
  * Open-Meteo Elevation API  https://api.open-meteo.com/v1/elevation (Copernicus DEM GLO-90, 90 м,
      до 100 точок за запит) → висота, ухил і експозиція (3×3 сітка, метод Горна).

  ҐРУНТИ
  * ISRIC SoilGrids 2.0 — REST https://rest.isric.org/soilgrids/v2.0/properties/query
      (станом на 2026 р. REST тимчасово призупинено ISRIC → автоматичний перехід на WCS);
      WCS https://maps.isric.org/mapserv?map=/map/<property>.map (GeoTIFF, потребує tifffile).
  * Вбудований довідник ґрунтів України (databases.SOILS_UA) — якщо API недоступні.
  * Публічна кадастрова карта (map.land.gov.ua) — шар «Агровиробничі групи ґрунтів» для ручної звірки.

  ГЕОЛОГІЯ
  * BGR IGME5000 (Міжнародна геологічна карта Європи 1:5 млн, покриває Україну), ArcGIS REST identify:
      https://services.bgr.de/arcgis/rest/services/inspire_ge/igme5000eu/MapServer/identify
  * Macrostrat https://macrostrat.org/api/v2/geologic_units/map?lat=..&lng=.. (глобальний резерв)

Усі відповіді кешуються на диску (TTL), щоб не перевищувати ліміти й працювати офлайн.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .databases import GEOLOGY

UA = "HydroBrain/1.0 (irrigation decision support; contact: project owner)"

HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m",
               "shortwave_radiation", "surface_pressure", "precipitation", "precipitation_probability",
               "et0_fao_evapotranspiration", "vapour_pressure_deficit", "cloud_cover",
               "soil_temperature_18cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"]


class SourceError(RuntimeError):
    pass


class DataSources:
    def __init__(self, cache_dir="hydro_data/cache", offline=False, timeout=20, commercial_api_key=None):
        self.cache_dir = cache_dir
        self.offline = offline
        self.timeout = timeout
        self.key = commercial_api_key
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _get(self, url, params, ttl_s=3600, binary=False, expect_tiff=False):
        key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key + (".bin" if binary else ".json"))
        if os.path.exists(path) and (self.offline or time.time() - os.path.getmtime(path) < ttl_s):
            with open(path, "rb" if binary else "r", encoding=None if binary else "utf-8") as f:
                return f.read() if binary else json.load(f)
        if self.offline:
            raise SourceError(f"офлайн-режим, немає кешу для {url}")
        if requests is None:
            raise SourceError("не встановлено пакет requests")
        try:
            r = requests.get(url, params=params, timeout=self.timeout, headers={"User-Agent": UA})
        except Exception as e:  # мережа
            raise SourceError(f"{url}: {e}") from e
        if r.status_code != 200:
            raise SourceError(f"{url}: HTTP {r.status_code} {r.text[:200]}")
        data = r.content if binary else r.json()
        if expect_tiff and not (data[:4] in (b"II*\x00", b"MM\x00*")):
            raise SourceError(f"{url}: замість GeoTIFF прийшло: {data[:160]!r}")
        with open(path, "wb" if binary else "w", encoding=None if binary else "utf-8") as f:
            if binary:
                f.write(data)
            else:
                json.dump(data, f)
        return data

    # ------------------------------------------------------------------
    # ПОГОДА
    # ------------------------------------------------------------------
    def forecast(self, lat, lon, past_days=7, forecast_days=7, models=None):
        """
        Погодинний ряд (UTC) від now−past_days до now+forecast_days.
        Повертає dict: times (list[datetime UTC]), і масиви змінних HOURLY_VARS (вітер у м/с).
        """
        params = dict(latitude=round(lat, 4), longitude=round(lon, 4), hourly=",".join(HOURLY_VARS),
                      past_days=int(min(max(past_days, 0), 92)), forecast_days=int(min(forecast_days, 16)),
                      wind_speed_unit="ms", timezone="GMT", timeformat="unixtime")
        if models:
            params["models"] = models
        if self.key:
            params["apikey"] = self.key
        data = self._get("https://api.open-meteo.com/v1/forecast", params, ttl_s=1800)
        return self._parse_hourly(data)

    def archive(self, lat, lon, start_date, end_date):
        """Історія ERA5/ERA5-Land (затримка ~5 діб) — для довгого «розгону» моделі від сівби."""
        params = dict(latitude=round(lat, 4), longitude=round(lon, 4), start_date=str(start_date),
                      end_date=str(end_date), hourly=",".join(v for v in HOURLY_VARS if "probability" not in v),
                      wind_speed_unit="ms", timezone="GMT", timeformat="unixtime")
        data = self._get("https://archive-api.open-meteo.com/v1/archive", params, ttl_s=7 * 86400)
        return self._parse_hourly(data)

    def era5land_soil(self, lat, lon, start_date, end_date):
        """
        Вологість ґрунту ERA5-Land (шари 0–7, 7–28, 28–100 см, м³/м³) — для оцінки стану поля без датчика
        і кліматології точки. Затримка даних ~5 діб. Повертає dict: times + sm1, sm2, sm3 (масиви).
        """
        params = dict(latitude=round(lat, 4), longitude=round(lon, 4), start_date=str(start_date),
                      end_date=str(end_date), models="era5_land", timezone="GMT", timeformat="unixtime",
                      hourly="soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm")
        data = self._get("https://archive-api.open-meteo.com/v1/archive", params, ttl_s=12 * 3600)
        hourly = data.get("hourly", {})
        times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in hourly.get("time", [])]
        f = lambda k: np.array([np.nan if v is None else v for v in hourly.get(k, [None] * len(times))], float)
        out = dict(times=times, sm1=f("soil_moisture_0_to_7cm"), sm2=f("soil_moisture_7_to_28cm"),
                   sm3=f("soil_moisture_28_to_100cm"))
        if not times or not np.isfinite(out["sm2"]).any():
            raise SourceError("ERA5-Land: немає даних про вологість ґрунту для цієї точки")
        return out

    def ensemble_precip(self, lat, lon, forecast_days=7, model="ecmwf_ifs025"):
        """
        Ансамбль опадів: масив (M, T) — M членів. Ключі відповіді: precipitation, precipitation_member01…
        """
        params = dict(latitude=round(lat, 4), longitude=round(lon, 4), hourly="precipitation",
                      models=model, forecast_days=int(min(forecast_days, 15)), timezone="GMT",
                      timeformat="unixtime")
        data = self._get("https://ensemble-api.open-meteo.com/v1/ensemble", params, ttl_s=3600)
        hourly = data.get("hourly", {})
        times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in hourly.get("time", [])]
        members = [np.asarray(v, dtype=float) for k, v in sorted(hourly.items()) if k.startswith("precipitation")]
        if not members:
            raise SourceError("ансамбль без членів precipitation*")
        arr = np.vstack(members)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return dict(times=times, members=arr)

    def nasa_power_daily(self, lat, lon, start, end):
        params = dict(parameters="T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR", community="AG",
                      longitude=round(lon, 4), latitude=round(lat, 4), start=start.strftime("%Y%m%d"),
                      end=end.strftime("%Y%m%d"), format="JSON")
        data = self._get("https://power.larc.nasa.gov/api/temporal/daily/point", params, ttl_s=7 * 86400)
        return data["properties"]["parameter"]

    @staticmethod
    def _parse_hourly(data):
        h = data.get("hourly")
        if not h or "time" not in h:
            raise SourceError("відповідь без блоку hourly")
        times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in h["time"]]
        out = {"times": times, "elevation": data.get("elevation")}
        for v in HOURLY_VARS:
            if v in h:
                out[v] = np.array([np.nan if x is None else x for x in h[v]], dtype=float)
        return out

    # ------------------------------------------------------------------
    # РЕЛЬЄФ
    # ------------------------------------------------------------------
    def terrain(self, lat, lon, spacing_m=90.0):
        """Висота (м), ухил (°) і експозиція (°, від півночі за год. стрілкою) за 3×3 сіткою DEM."""
        dlat = spacing_m / 111320.0
        dlon = spacing_m / (111320.0 * math.cos(math.radians(lat)))
        lats, lons = [], []
        for r in (1, 0, -1):          # північ → південь
            for c in (-1, 0, 1):      # захід → схід
                lats.append(round(lat + r * dlat, 6))
                lons.append(round(lon + c * dlon, 6))
        data = self._get("https://api.open-meteo.com/v1/elevation",
                         dict(latitude=",".join(map(str, lats)), longitude=",".join(map(str, lons))),
                         ttl_s=365 * 86400)
        z = np.array(data["elevation"], dtype=float).reshape(3, 3)
        # метод Горна
        dzdx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * spacing_m)
        dzdy = ((z[0, 0] + 2 * z[0, 1] + z[0, 2]) - (z[2, 0] + 2 * z[2, 1] + z[2, 2])) / (8 * spacing_m)
        slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
        aspect = (math.degrees(math.atan2(-dzdx, -dzdy)) + 360.0) % 360.0  # напрям, куди «дивиться» схил
        return dict(alt=float(z[1, 1]), slope_deg=round(slope, 2), aspect_deg=round(aspect, 1))

    # ------------------------------------------------------------------
    # ҐРУНТИ (SoilGrids)
    # ------------------------------------------------------------------
    SG_PROPS = {"sand": 10.0, "clay": 10.0, "silt": 10.0, "soc": 10.0, "bdod": 100.0, "phh2o": 10.0,
                "cfvo": 10.0}  # ділення: % / g/kg / g/cm3 / pH / vol%
    SG_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm"]
    SG_W = np.array([5, 10, 15, 30], dtype=float)

    def soilgrids(self, lat, lon, depth_cm=60):
        """
        Зважені за товщиною шарів властивості 0-60 см. Спершу REST, потім WCS.
        SoilGrids не має прогнозів для забудови, води та деяких інших масок — тому, якщо в самій
        точці порожньо, береться найближча ділянка з даними (до ~2 км; зазвичай це сусіднє поле).
        """
        errs = []
        try:
            return self._soilgrids_rest(lat, lon)
        except Exception as e:
            errs.append(f"REST: {e}")
        try:
            return self._soilgrids_wcs(lat, lon)
        except Exception as e:
            errs.append(f"WCS: {e}")
        raise SourceError("SoilGrids недоступний (" + "; ".join(errs) + ")")

    def _sg_rest_point(self, lat, lon):
        params = [("lon", round(lon, 5)), ("lat", round(lat, 5)), ("value", "mean")]
        params += [("property", p) for p in self.SG_PROPS] + [("depth", d) for d in self.SG_DEPTHS]
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        # повторювані ключі (property=…&property=…) передаємо списком кортежів, тому запит — напряму
        key = hashlib.sha1((url + json.dumps(params)).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        elif self.offline or requests is None:
            raise SourceError("офлайн / немає requests")
        else:
            try:
                r = requests.get(url, params=params, timeout=self.timeout, headers={"User-Agent": UA})
            except Exception as e:
                raise SourceError(str(e))
            if r.status_code != 200:
                raise SourceError(f"HTTP {r.status_code}")
            data = r.json()
        out = {}
        for layer in data.get("properties", {}).get("layers", []):
            name = layer["name"]
            div = layer.get("unit_measure", {}).get("d_factor", self.SG_PROPS.get(name, 1.0))
            vals = []
            for d in layer["depths"]:
                v = d["values"].get("mean")
                vals.append(np.nan if v is None else v / div)
            vals = np.array(vals[:4], dtype=float)
            if np.all(np.isnan(vals)):
                continue
            w = self.SG_W[:len(vals)][~np.isnan(vals)]
            out[name] = float(np.sum(vals[~np.isnan(vals)] * w) / w.sum())
        if "sand" in out and "clay" in out and not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:   # кешуємо лише непорожні відповіді
                json.dump(data, f)
        return out

    def _soilgrids_rest(self, lat, lon):
        # сама точка, потім 4 сусідні ~1 км (ліміт ISRIC — 5 запитів/хв, тому не більше)
        dlat = 0.009
        dlon = 0.009 / max(0.2, math.cos(math.radians(lat)))
        pts = [(lat, lon), (lat + dlat, lon), (lat - dlat, lon), (lat, lon + dlon), (lat, lon - dlon)]
        for k, (la, lo) in enumerate(pts):
            out = self._sg_rest_point(la, lo)
            if "sand" in out and "clay" in out:
                res = self._sg_finalize(out, "SoilGrids REST")
                if k > 0:
                    res["note"] = "у самій точці даних немає (забудова/вода) — взято сусідню ділянку ~1 км"
                return res
        raise SourceError("немає даних у точці і поруч (ймовірно, забудова або водойма)")

    def _soilgrids_wcs(self, lat, lon, half_deg=0.02):
        try:
            import io
            import tifffile
        except ImportError:
            raise SourceError("для WCS потрібен пакет tifffile (pip install tifffile)")
        out, used_far = {}, False
        d = half_deg
        for prop, div in self.SG_PROPS.items():
            vals = []
            for depth in self.SG_DEPTHS:
                cov = f"{prop}_{depth}_mean"
                params = {"map": f"/map/{prop}.map", "SERVICE": "WCS", "VERSION": "2.0.1",
                          "REQUEST": "GetCoverage", "COVERAGEID": cov, "FORMAT": "image/tiff",
                          "SUBSET": [f"long({lon - d},{lon + d})", f"lat({lat - d},{lat + d})"],
                          "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
                          "OUTPUTCRS": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                raw = self._get("https://maps.isric.org/mapserv", params, ttl_s=365 * 86400, binary=True,
                                expect_tiff=True)
                arr = np.asarray(tifffile.imread(io.BytesIO(raw)), dtype=float)
                if arr.ndim > 2:
                    arr = arr[..., 0] if arr.shape[-1] < arr.shape[0] else arr[0]
                v, far = self._nearest_valid(arr)
                used_far |= far
                vals.append(np.nan if v is None else v / div)
            vals = np.array(vals)
            if np.all(np.isnan(vals)):
                continue
            w = self.SG_W[~np.isnan(vals)]
            out[prop] = float(np.sum(vals[~np.isnan(vals)] * w) / w.sum())
        res = self._sg_finalize(out, "SoilGrids WCS")
        if used_far:
            res["note"] = "у самій точці даних немає (забудова/вода) — взято найближчі пікселі з даними"
        return res

    @staticmethod
    def _nearest_valid(arr, k=9):
        """Медіана k найближчих до центру пікселів із даними (nodata SoilGrids = −32768)."""
        if arr.ndim != 2 or arr.size == 0:
            return None, False
        ok = (arr > -32000) & (arr != 0) & np.isfinite(arr)
        if not ok.any():
            return None, False
        cy, cx = (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0
        yy, xx = np.nonzero(ok)
        dist = np.hypot(yy - cy, xx - cx)
        order = np.argsort(dist)[:k]
        far = bool(dist[order[0]] > 1.5)
        return float(np.median(arr[yy[order], xx[order]])), far

    @staticmethod
    def _sg_finalize(out, src):
        if "sand" not in out or "clay" not in out:
            raise SourceError("немає піску/глини")
        res = dict(sand=out["sand"], clay=out["clay"], source=src)
        if res["sand"] + res["clay"] > 100 or res["sand"] < 0 or res["clay"] < 0:
            raise SourceError(f"неправдоподібні значення піску/глини: {res}")
        if "soc" in out:
            res["om"] = out["soc"] / 10.0 * 1.724      # g/kg → % C → % OM (коеф. ван Беммелена)
        if "bdod" in out:
            res["bd"] = out["bdod"]
        if "phh2o" in out:
            res["ph"] = out["phh2o"]
        if "cfvo" in out:
            res["stones"] = out["cfvo"] / 100.0
        return res

    # ------------------------------------------------------------------
    # ГЕОЛОГІЯ
    # ------------------------------------------------------------------
    def geology(self, lat, lon):
        """Опис корінних порід → клас GEOLOGY. Спершу IGME5000 (BGR), потім Macrostrat."""
        texts, src, errs = [], None, []
        try:
            params = dict(geometry=json.dumps({"x": lon, "y": lat}), geometryType="esriGeometryPoint", sr=4326,
                          layers="all:2,3", tolerance=2, mapExtent=f"{lon - 0.05},{lat - 0.05},{lon + 0.05},{lat + 0.05}",
                          imageDisplay="400,400,96", returnGeometry="false", f="json")
            data = self._get("https://services.bgr.de/arcgis/rest/services/inspire_ge/igme5000eu/MapServer/identify",
                             params, ttl_s=365 * 86400)
            for r in data.get("results", []):
                texts += [str(v) for v in r.get("attributes", {}).values() if isinstance(v, str)]
            src = "BGR IGME5000"
        except SourceError as e:
            errs.append(f"IGME5000: {e}")
        if not texts:
            try:
                data = self._get("https://macrostrat.org/api/v2/geologic_units/map",
                                 dict(lat=lat, lng=lon, format="json"), ttl_s=365 * 86400)
                for u in data.get("success", {}).get("data", []):
                    texts += [str(u.get(k, "")) for k in ("name", "strat_name", "lith", "descrip", "b_int_name")]
                src = "Macrostrat"
            except SourceError as e:
                errs.append(f"Macrostrat: {e}")
        if not texts:
            raise SourceError("геологічні сервіси не дали опису породи (" + "; ".join(errs)[:300] + ")")
        blob = " ".join(texts).lower()
        best, score = "unknown", 0
        for k, g in GEOLOGY.items():
            s = sum(blob.count(w) for w in g["keywords"])
            if s > score:
                best, score = k, s
        return dict(geo=best, description=" | ".join(t for t in texts if t)[:400], source=src)
