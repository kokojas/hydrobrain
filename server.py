"""
HTTP-обгортка «мозку» для підключення до вже готового інтерфейсу (будь-яка мова/фреймворк).

Запуск:
    pip install fastapi uvicorn
    uvicorn server:app --host 0.0.0.0 --port 8000
Простий інтерфейс у браузері:      http://localhost:8000/
Документація API з'явиться автоматично: http://localhost:8000/docs
Демо без інтернету (синтетична погода):  HYDRO_OFFLINE=1 uvicorn server:app
Поле без датчика (1.8):  HYDRO_NO_SENSOR_POLICY=orientative (за замовчуванням, орієнтовна рекомендація)
                         або HYDRO_NO_SENSOR_POLICY=off (без датчика — лише прохання зробити вимір)

Усі відповіді — JSON. Помилки зовнішніх сервісів (погода/карти) не «валять» запит:
вони повертаються в полі notes/warnings, а система бере довідникові значення.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hydrobrain import HydroBrain, __version__

DATA_DIR = os.environ.get("HYDRO_DATA", "hydro_data")
OFFLINE = os.environ.get("HYDRO_OFFLINE") == "1"
brain = HydroBrain(DATA_DIR, offline=OFFLINE)
UI = Path(__file__).resolve().parent / "ui" / "index.html"
app = FastAPI(title="HydroBrain — мозок поливу", version=__version__)
app.mount("/vendor", StaticFiles(directory=str(UI.parent / "vendor")), name="vendor")   # Leaflet локально
# дозволяє відкривати ui/index.html і як файл (file://), і з іншого порту/інтерфейсу
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(FileNotFoundError)
async def field_not_found(request, exc):
    return JSONResponse(status_code=404, content={"detail": "Поле не знайдено"})


@app.exception_handler(ValueError)
async def invalid_input(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


class FieldIn(BaseModel):
    inputs: dict[str, Any]
    mode: str = "beginner"          # beginner | expert
    name: Optional[str] = None


class SensorIn(BaseModel):
    theta: float                     # м³/м³ або %
    depth_cm: Optional[float] = None
    time: Optional[str] = None       # ISO 8601; без зони = Київ
    sensor_id: str = "s1"


class IrrigationIn(BaseModel):
    amount_mm: Optional[float] = None
    volume_m3: Optional[float] = None
    time: Optional[str] = None
    recommended_mm: Optional[float] = None
    note: Optional[str] = None
    cycles: Optional[bool] = None    # 1.10: поливали циклами з паузами (None — за останньою рекомендацією)


class FertIn(BaseModel):
    dose_kg_ha: float
    fert_type: str = "npk_complex"
    time: Optional[str] = None


@app.get("/schema/{mode}")
def schema(mode: str = "beginner"):
    """Поля форми для режиму beginner/expert (назви, одиниці, межі, варіанти вибору)."""
    return brain.input_schema(mode)


@app.get("/catalog")
def catalog():
    return brain.catalog()


@app.post("/fields")
def create_field(body: FieldIn):
    try:
        return brain.create_field(body.inputs, mode=body.mode, name=body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/fields/{fid}")
def update_field(fid: str, overrides: dict[str, Any], mode: Optional[str] = None):
    return brain.update_params(fid, overrides, mode)


@app.get("/fields")
def list_fields():
    return brain.list_fields()


@app.post("/fields/{fid}/sensor")
def add_sensor(fid: str, body: SensorIn):
    brain.add_sensor_reading(fid, body.theta, body.depth_cm, body.time, body.sensor_id)
    return {"ok": True}


@app.post("/fields/{fid}/irrigation")
def add_irrigation(fid: str, body: IrrigationIn):
    try:
        return brain.log_irrigation(fid, body.amount_mm, body.time, body.recommended_mm, body.volume_m3, body.note,
                                    body.cycles)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/fields/{fid}/calibrate_soil")
def calibrate_soil(fid: str):
    """Калібрує θFC/θWP за накопиченими показами датчика (hydrobrain/soil_calibration.py)."""
    try:
        with brain.lock(fid):
            return brain.calibrate_soil(fid)
    except FileNotFoundError:
        raise HTTPException(404, "Поле не знайдено")


@app.post("/fields/{fid}/finetune_local")
def finetune_local(fid: str):
    """Локальне донавчання ML-ядра на даних цього поля (після ~сезону спостережень)."""
    try:
        with brain.lock(fid):
            return brain.finetune_local(fid)
    except FileNotFoundError:
        raise HTTPException(404, "Поле не знайдено")


@app.post("/fields/{fid}/fertilizer")
def add_fert(fid: str, body: FertIn):
    brain.add_fertilizer(fid, body.dose_kg_ha, body.fert_type, body.time)
    return {"ok": True}


@app.get("/", include_in_schema=False)
def ui():
    """Простий веб-інтерфейс (ui/index.html) — працює поверх цих самих ендпоінтів."""
    return FileResponse(UI)


@app.get("/fields/{fid}/info")
def field_info(fid: str):
    """Коротка інформація про поле для інтерфейсу (лише читання)."""
    try:
        f = brain.load_field(fid)
    except FileNotFoundError:
        raise HTTPException(404, "Поле не знайдено")
    P = f["params"]
    return dict(field_id=fid, name=f.get("name"), mode=f.get("mode"), warnings=f.get("warnings", []),
                soil_hydraulics=f.get("soil_hydraulics"), offline_demo=OFFLINE,
                summary={k: P.get(k) for k in ("lat", "lon", "crop", "sowing_date", "irrigation_method",
                                               "soil_type_ua", "geo", "alt", "slope_deg", "aspect_deg",
                                               "sand", "clay", "om", "area_m2", "flow_rate_m3_h")},
                events={k: v[-10:] for k, v in f.get("events", {}).items()},
                last_recommendation={k: v for k, v in (f.get("last_recommendation") or {}).items() if k != "ctx"})


@app.get("/fields_detailed")
def list_fields_detailed():
    out = []
    for fid in brain.list_fields():
        try:
            f = brain.load_field(fid)
            out.append(dict(field_id=fid, name=f.get("name"), crop=f["params"].get("crop"),
                            created=f.get("created")))
        except Exception:
            pass
    return sorted(out, key=lambda d: d.get("created") or "", reverse=True)


def _demo_weather(fid):
    """Синтетична погода для режиму без інтернету (HYDRO_OFFLINE=1)."""
    from hydrobrain.synthetic import make_weather
    P = brain.load_field(fid)["params"]
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hours = 62 * 24 + 120
    return make_weather(P["lat"], P["lon"], now - timedelta(days=62), hours, tmax=28, tmin=15,
                        rain_events=[(62 * 24 + 30, 3.0, 4, 75)])


def _recommend(fid):
    with brain.lock(fid):      # v1.2: одночасні запити до одного поля виконуються по черзі
        if OFFLINE:
            return brain.recommend(fid, weather=_demo_weather(fid))
        return brain.recommend(fid)


@app.post("/fields/{fid}/recommendation")
def recommend_post(fid: str):
    """Рекомендація (v1.2, правильний метод): запит змінює збережений стан поля, тому це POST."""
    try:
        return _recommend(fid)
    except FileNotFoundError:
        raise HTTPException(404, "Поле не знайдено")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(503, f"Не вдалося отримати дані: {e}")


@app.get("/fields/{fid}/recommendation", deprecated=True)
def recommend(fid: str):
    """Залишено для сумісності з інтерфейсом версій ≤ 1.1; новий код має використовувати POST."""
    try:
        return _recommend(fid)
    except FileNotFoundError:
        raise HTTPException(404, "Поле не знайдено")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(503, f"Не вдалося отримати дані: {e}")
