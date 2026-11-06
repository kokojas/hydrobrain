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
from functools import lru_cache
from typing import Any, Optional

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from hydrobrain import HydroBrain, __version__
from hydrobrain.sessions import SessionStore

DATA_DIR = os.environ.get("HYDRO_DATA", "hydro_data")
OFFLINE = os.environ.get("HYDRO_OFFLINE") == "1"
brain = HydroBrain(DATA_DIR, offline=OFFLINE)
UI = Path(__file__).resolve().parent / "ui" / "index.html"
app = FastAPI(title="HydroBrain — мозок поливу", version=__version__)
app.mount("/vendor", StaticFiles(directory=str(UI.parent / "vendor")), name="vendor")   # Leaflet локально
# дозволяє відкривати ui/index.html і як файл (file://), і з іншого порту/інтерфейсу
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=4)
def _session_store(data_dir):
    return SessionStore(data_dir)


def session_store():
    return _session_store(str(Path(brain.data_dir).resolve()))


def require_session(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    owner = session_store().authenticate(credentials.credentials) if credentials else None
    if owner is None:
        raise HTTPException(401, "Не вдалося підтвердити сесію браузера. Оновіть сторінку.",
                            headers={"WWW-Authenticate": "Bearer"})
    return owner


def owned_field(fid: str, owner: str = Depends(require_session)):
    # Unknown IDs and another visitor's IDs have the same response.
    if not session_store().owns(fid, owner):
        raise HTTPException(404, "Поле не знайдено")
    return fid


@app.middleware("http")
async def private_responses(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/fields", "/sessions")):
        response.headers["Cache-Control"] = "private, no-store"
        vary = response.headers.get("Vary", "")
        response.headers["Vary"] = ", ".join(filter(None, (vary, "Authorization")))
    return response


@app.post("/sessions")
def create_session():
    return {"token": session_store().create(), "token_type": "Bearer"}


@app.get("/sessions/current")
def current_session(owner: str = Depends(require_session)):
    return {"ok": True}


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


@app.get("/healthz")
def health():
    return {"status": "ok", "version": __version__, "offline_demo": OFFLINE,
            "ml_available": brain.ml_core is not None}


@app.get("/catalog")
def catalog():
    return brain.catalog()


@app.post("/fields")
def create_field(body: FieldIn, owner: str = Depends(require_session)):
    try:
        field = brain.create_field(body.inputs, mode=body.mode, name=body.name)
        session_store().claim_field(field["field_id"], owner)
        return field
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/fields/{fid}")
def update_field(overrides: dict[str, Any], fid: str = Depends(owned_field), mode: Optional[str] = None):
    return brain.update_params(fid, overrides, mode)


@app.get("/fields")
def list_fields(owner: str = Depends(require_session)):
    return session_store().fields(owner)


@app.post("/fields/{fid}/sensor")
def add_sensor(body: SensorIn, fid: str = Depends(owned_field)):
    brain.add_sensor_reading(fid, body.theta, body.depth_cm, body.time, body.sensor_id)
    return {"ok": True}


@app.post("/fields/{fid}/irrigation")
def add_irrigation(body: IrrigationIn, fid: str = Depends(owned_field)):
    try:
        return brain.log_irrigation(fid, body.amount_mm, body.time, body.recommended_mm, body.volume_m3, body.note,
                                    body.cycles)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/fields/{fid}/calibrate_soil")
def calibrate_soil(fid: str = Depends(owned_field)):
    """Калібрує θFC/θWP за накопиченими показами датчика (hydrobrain/soil_calibration.py)."""
    try:
        with brain.lock(fid):
            return brain.calibrate_soil(fid)
    except FileNotFoundError:
        raise HTTPException(404, "Поле не знайдено")


@app.post("/fields/{fid}/finetune_local")
def finetune_local(fid: str = Depends(owned_field)):
    """Локальне донавчання ML-ядра на даних цього поля (після ~сезону спостережень)."""
    try:
        with brain.lock(fid):
            return brain.finetune_local(fid)
    except FileNotFoundError:
        raise HTTPException(404, "Поле не знайдено")


@app.post("/fields/{fid}/fertilizer")
def add_fert(body: FertIn, fid: str = Depends(owned_field)):
    brain.add_fertilizer(fid, body.dose_kg_ha, body.fert_type, body.time)
    return {"ok": True}


@app.get("/", include_in_schema=False)
def ui():
    """Простий веб-інтерфейс (ui/index.html) — працює поверх цих самих ендпоінтів."""
    return FileResponse(UI, headers={"Cache-Control": "no-cache"})


@app.get("/session.js", include_in_schema=False)
def session_script():
    return FileResponse(UI.parent / "session.js", media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/fields/{fid}/info")
def field_info(fid: str = Depends(owned_field)):
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
def list_fields_detailed(owner: str = Depends(require_session)):
    out = []
    for fid in session_store().fields(owner):
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
def recommend_post(fid: str = Depends(owned_field)):
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
def recommend(fid: str = Depends(owned_field)):
    """Залишено для сумісності з інтерфейсом версій ≤ 1.1; новий код має використовувати POST."""
    try:
        return _recommend(fid)
    except FileNotFoundError:
        raise HTTPException(404, "Поле не знайдено")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(503, f"Не вдалося отримати дані: {e}")
