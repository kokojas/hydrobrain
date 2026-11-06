"""
Ядро: погодинний водний баланс кореневої зони (FAO-56, гл. 7-8, подвійний Kc),
розширений параметрами зі схеми проєкту.

Рівняння (у термінах дефіциту Dr, мм; Dr = 0 — польова вологоємність):

  Dr[t+1] = Dr[t] − (P·K_slope) − (I·Ei·K_sch) − CR(Geo, GW)
            + ETo_loc·(Ks·Kstress·Kcb·Kx + Ke) + DP(Dr) + ...

  ETo_loc   — ETo з мікрокліматом (ухил/експозиція, затінення, вентиляція, альбедо, CO2) = ETo·K_micro
  K_slope   — частка опадів, що увійшла в ґрунт: 1 − (RO_CN + перехоплення)/P
  Ei·K_sch  — ефективність способу поливу з поправкою на вітер/час доби (знос крапель)
  Ks        — водний стрес (FAO-56, рівн. 84); Kstress — солі (рівн. 91), pH, Tґрунту, Tводи
  Kx        — густота посіву, здоров'я, продихи, мульча; Ke — випаровування з ґрунту (рівн. 71)
  DP        — дренаж надлишку понад FC зі швидкістю τ (клас дренажу)

Модель векторизована: стан — масиви довжини N (N сценаріїв Монте-Карло; N = 1 для «історії»).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from . import crop as cropm
from . import meteo, salinity, soil
from .databases import CROPS, FERTILIZERS, IRRIGATION_METHODS


Z_UP = 0.30   # м — товщина верхнього шару кореневої зони (типова глибина датчиків 10–30 см)

# Сніг — значення за замовчуванням SWAT+ (snow.sno): поріг снігопаду 0 °C, поріг танення 0.5 °C,
# коефіцієнт танення 4.5 (21 червня) … 0.5 (21 грудня) мм/°C/добу, повне покриття при SWE ≥ 25 мм.
SNOW = dict(falltmp=0.0, melttmp=0.5, meltmx=4.5, meltmn=0.5, covmx=25.0)


def root_share(x):
    """Частка коренів у верхній частині x (0..1) кореневої зони — правило 40-30-20-10 (по чвертях глибини)."""
    return float(np.interp(np.clip(x, 0.0, 1.0), [0, 0.25, 0.5, 0.75, 1.0], [0, 0.4, 0.7, 0.9, 1.0]))


def moving_mean(x, w):
    """
    Центроване ковзне середнє, коректне на краях ряду: ділимо на кількість реально наявних точок.
    (np.convolve(..., mode="same")/w доповнює краї нулями — у версіях ≤ 1.1 це штучно занижувало
    температуру ґрунту в перші й останні ~w/2 годин, тобто й транспірацію в кінці горизонту прогнозу.)
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    ok = np.isfinite(x)
    k = np.ones(int(max(1, min(w, x.size))))   # для рядів, коротших за вікно (інакше «same» поверне довший масив)
    num = np.convolve(np.where(ok, x, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


@dataclass
class Series:
    """Погодинні ряди, що не залежать від сценарію (готуються один раз)."""
    times: list
    eto: np.ndarray          # ETo локальна, мм/год
    eto_station: np.ndarray  # ETo без мікроклімату
    k_micro: np.ndarray
    kcb: np.ndarray
    kcmax: np.ndarray
    fc: np.ndarray
    lai: np.ndarray
    h: np.ndarray
    zr: np.ndarray
    p: np.ndarray
    kx: np.ndarray           # добуток додаткових множників транспірації
    dvs: np.ndarray
    day_index: np.ndarray    # номер локальної доби для кожної години
    day_starts: np.ndarray   # індекси початку діб
    u2: np.ndarray
    rh: np.ndarray
    t_air: np.ndarray
    is_day: np.ndarray
    drift: np.ndarray        # додаткові втрати на знос/випаровування крапель (частка)
    fert_dece: np.ndarray    # приріст ECe від добрив у годину t
    local_hour: np.ndarray
    extra: dict = field(default_factory=dict)


class WaterBalanceEngine:
    def __init__(self, P: dict):
        """P — розв'язані параметри поля (див. params.resolve)."""
        self.P = P
        self.crop = dict(CROPS[P["crop"]])
        if P.get("kc_mid_override"):
            self.crop["kcb_mid"] = float(P["kc_mid_override"])
        if P.get("zr_max_override"):
            self.crop["zr"] = (min(self.crop["zr"][0], float(P["zr_max_override"])), float(P["zr_max_override"]))
        self.method = dict(IRRIGATION_METHODS[P["irrigation_method"]])
        hyd = soil.saxton_rawls(P["sand"], P["clay"], P["om"], P["bd"], P.get("stones", 0.0))
        g = float(np.clip(P.get("stones", 0.0) or 0.0, 0.0, 0.8))
        if P.get("theta_fc") is not None:      # виміряні/довідникові — для дрібнозему; каміння зменшує об'єм
            hyd["theta_fc"] = float(P["theta_fc"]) * (1 - g)
        if P.get("theta_wp") is not None:
            hyd["theta_wp"] = float(P["theta_wp"]) * (1 - g)
        if P.get("theta_sat") is not None:
            hyd["theta_sat"] = float(P["theta_sat"])
        hyd["theta_sat"] = max(hyd["theta_sat"], hyd["theta_fc"] + 0.03)
        ks_mult, cn_struct = soil.structure_effects(P.get("structure", "medium"))
        hyd["ksat_mm_h"] *= ks_mult
        if P.get("ksat_mm_h") is not None:
            hyd["ksat_mm_h"] = float(P["ksat_mm_h"])
        self.hyd = hyd
        self.tex = soil.texture_class(P["sand"], P["clay"])
        self.tew, self.rew = soil.tew_rew(hyd["theta_fc"], hyd["theta_wp"], self.tex)
        self.tau = soil.drainage_tau(hyd["ksat_mm_h"], P.get("drainage_class", "good"))
        self.tau_h = 1.0 - (1.0 - self.tau) ** (1.0 / 24.0)
        self.hsg = P.get("hsg") or soil.hydrologic_group(hyd["ksat_mm_h"], P.get("ds", 1.5), P.get("gw_depth"))
        cover = self.crop["cover"]
        if P.get("mulch_type") in ("organic",) and cover == "row":
            cover = "row_residue"
        cn2 = soil.cn_base(cover, self.hsg)
        from .databases import DRAINAGE_CLASSES
        cn2 += cn_struct + DRAINAGE_CLASSES.get(P.get("drainage_class", "good"), {}).get("cn_shift", 0)
        cn2 += 8.0 * max(0.0, hyd["density_factor"] - 1.0) / 0.3   # ущільнення → більше стоку
        self.cn2 = float(np.clip(soil.cn_slope_adjust(cn2, P.get("slope_deg", 0.0) * np.pi / 180.0), 30, 98))
        self.taw_unit = 1000.0 * (hyd["theta_fc"] - hyd["theta_wp"])   # мм на 1 м коренів
        self.sat_unit = 1000.0 * (hyd["theta_sat"] - hyd["theta_fc"])
        self.ece_base = salinity.ec_base(P["ece_soil"], P.get("geo_ece_add", 0.0), P["ph"])
        prev = P.get("prev_crop", "unknown")
        frac = {"fallow": 0.1, "sunflower": 0.7, "maize": 0.5, "sugar_beet": 0.6, "winter_wheat": 0.5,
                "lucerne": 0.75, "unknown": 0.3}.get(prev, 0.3)
        self.theta_sub = hyd["theta_fc"] - frac * (hyd["theta_fc"] - hyd["theta_wp"])
        self.cr_on = P.get("gw_depth") is not None

    # ------------------------------------------------------------------
    def prepare(self, times_utc: list, wx: dict, dvs: np.ndarray, soil_temp=None,
                fert_events=None, tz_offset_h=3.0) -> Series:
        P, crop = self.P, self.crop
        micro = dict(slope_deg=P.get("slope_deg", 0.0), aspect_deg=P.get("aspect_deg", 180.0),
                     shade=P.get("shade", 0.0), vent=P.get("vent", "normal"),
                     albedo=P.get("albedo", 0.23), co2_ppm=P.get("co2_ppm"))
        met = meteo.compute_local_eto_series(times_utc, wx, P["lat"], P["lon"], P["alt"], micro)
        n = len(times_utc)
        local = [t + timedelta(hours=tz_offset_h) for t in times_utc]
        dates = np.array([lt.date().toordinal() for lt in local])
        day_index = dates - dates[0]
        day_starts = np.r_[0, np.nonzero(np.diff(day_index))[0] + 1]
        local_hour = np.array([lt.hour for lt in local])
        # добові RHmin і u2 для кліматичної поправки Kc
        rhmin_h = np.zeros(n)
        u2_day = np.zeros(n)
        for k, s in enumerate(day_starts):
            e = day_starts[k + 1] if k + 1 < len(day_starts) else n
            rhmin_h[s:e] = np.min(met["rh"][s:e])
            u2_day[s:e] = np.mean(met["u2_local"][s:e])
        h = cropm.height_curve(dvs, crop)
        kmid = cropm.climate_adjust(crop["kcb_mid"], u2_day, rhmin_h, crop["h"])
        kend = cropm.climate_adjust(crop["kcb_end"], u2_day, rhmin_h, crop["h"])
        kcb = cropm.kcb_curve(dvs, crop, kmid, kend)
        kcmax = cropm.kc_max(kcb, u2_day, rhmin_h, h)
        if P.get("lai_measured"):
            ratio = float(cropm.kcb_from_lai(P["lai_measured"], np.max(kmid)) / max(kcb[-1], 0.2))
            kcb = cropm.KC_MIN + (kcb - cropm.KC_MIN) * np.clip(ratio, 0.5, 1.5)
        fc = cropm.fraction_cover(kcb, kcmax, h)
        # густота посіву (FAO-56 рівн. 98)
        pd, pd_opt = P.get("plant_density"), P.get("plant_density_opt")
        if pd and pd_opt:
            fc_eff = fc * min(1.0, pd / pd_opt)
            kd_full = cropm.density_coefficient(np.maximum(fc, 1e-3), h)
            kd_eff = cropm.density_coefficient(fc_eff, h)
            kcb = cropm.KC_MIN + (kcb - cropm.KC_MIN) * np.where(kd_full > 0, kd_eff / kd_full, 1.0)
            fc = fc_eff
        # здоров'я рослин (0..1)
        kcb = cropm.KC_MIN + (kcb - cropm.KC_MIN) * float(P.get("plant_health", 1.0))
        # пластикова мульча зменшує Kcb на ~10% (FAO-56, гл. 11)
        if P.get("mulch_type") == "plastic":
            kcb = kcb * 0.90
        lai = P.get("lai_measured") or cropm.lai_from_fc(fc)
        lai = np.full(n, lai) if np.isscalar(lai) else lai
        ds_lim = min(P.get("ds", 1.5), P.get("ds_max", 3.0))
        if P.get("gw_depth") is not None:
            ds_lim = min(ds_lim, max(0.3, P["gw_depth"] - 0.3))
        zr = cropm.root_depth(dvs, crop, ds_lim)
        # p з поправкою на добову ETc
        # ETc за добу — ковзна 24-годинна сума (неповні доби на краях ряду не спотворюють p)
        etc_h = met["eto_local"] * kcb
        if n >= 24:
            roll = np.convolve(etc_h, np.ones(24), mode="valid")          # довжина n-23
            pad_l = (n - len(roll)) // 2
            etc_day = np.r_[np.full(pad_l, roll[0]), roll, np.full(n - len(roll) - pad_l, roll[-1])]
        else:
            etc_day = np.full(n, etc_h.mean() * 24.0)
        p = cropm.p_adjusted(crop["p"], etc_day, dvs, crop, P.get("mad_factor", 1.0))
        # інші множники транспірації
        if soil_temp is None:
            ts = moving_mean(met["t"], 72)
        else:
            ts = np.asarray(soil_temp, dtype=float)
            ts = np.where(np.isfinite(ts), ts, met["t"])
        kx = (cropm.k_soil_temperature(ts, crop["season"])
              * cropm.k_ph(P["ph"], crop["ph_opt"])
              * cropm.k_stomatal(P.get("gs"), P.get("gs_ref")))
        # знос/випаровування крапель при дощуванні (Playán et al., 2005), понад закладене в Ea
        if self.method["wind_sensitive"]:
            wdel = 20.34 + 0.214 * met["u2_local"] ** 2 - 0.229 * met["rh"]
            wdel = np.where(met["is_day"], wdel, 0.45 * wdel)
            ref = 20.34 + 0.214 * 4.0 - 0.229 * 60.0
            drift = np.clip((wdel - ref) / 100.0, -0.05, 0.35)
        else:
            drift = np.zeros(n)
        # добрива
        fert = np.zeros(n)
        for ev in fert_events or []:
            t_ev, dose, kind = ev["time"], ev["dose_kg_ha"], ev.get("type", "npk_complex")
            if times_utc[0] <= t_ev <= times_utc[-1]:
                import bisect
                i = min(bisect.bisect_left(times_utc, t_ev), n - 1)
                wfc = 1000.0 * self.hyd["theta_fc"] * zr[i]
                fert[i] += salinity.fert_delta_ece(dose, FERTILIZERS.get(kind, FERTILIZERS["npk_complex"])["si"], wfc)
        return Series(times=times_utc, eto=met["eto_local"], eto_station=met["eto_station"],
                      k_micro=met["k_micro"], kcb=kcb, kcmax=kcmax, fc=fc, lai=lai, h=h, zr=zr, p=p,
                      kx=kx, dvs=dvs, day_index=day_index, day_starts=day_starts, u2=met["u2_local"],
                      rh=met["rh"], t_air=met["t"], is_day=met["is_day"], drift=drift, fert_dece=fert,
                      local_hour=local_hour, extra=dict(ts=ts, rhmin=rhmin_h, etc_day=etc_day,
                                                        k_rad=met["k_rad"],
                                                        doy=np.array([lt.timetuple().tm_yday for lt in local])))

    # ------------------------------------------------------------------
    def init_state(self, n: int, dr0, de0=None, x0=0.0, zr0=None, du0=None, swe0=0.0):
        z = zr0 if zr0 is not None else self.crop["zr"][0]
        dr = np.full(n, float(dr0)) if np.isscalar(dr0) else np.asarray(dr0, float).copy()
        if du0 is None:   # без окремої інформації — верхній шар має таку саму відносну вологість, як уся зона
            du = np.clip(dr, 0, None) * min(Z_UP, z) / max(z, 1e-6)
        else:
            du = np.full(n, float(du0)) if np.isscalar(du0) else np.asarray(du0, float).copy()
        return dict(dr=dr, du=du,
                    de=np.full(n, self.rew if de0 is None else float(de0)),
                    x=np.full(n, float(x0)), fw=np.ones(n), zr=float(z),
                    swe=np.full(n, float(swe0)) if np.isscalar(swe0) else np.asarray(swe0, float).copy(),
                    h_since_irr=np.full(n, 999.0))

    def taw(self, zr, taw_mult=1.0):
        return self.taw_unit * zr * taw_mult

    def infiltration_fraction(self):
        """Частка поливу, що вбирається при безперервному поливі (1.2): Ksat / інтенсивність для дощування."""
        rate = self.application_rate()
        meth = IRRIGATION_METHODS[self.P["irrigation_method"]]
        if rate and meth["wind_sensitive"] and rate > self.hyd["ksat_mm_h"]:
            return float(self.hyd["ksat_mm_h"] / rate)
        return 1.0

    def application_rate(self):
        """Інтенсивність поливу, мм/год: явно задана або витрата системи / площа."""
        P = self.P
        if P.get("application_rate_mm_h"):
            return float(P["application_rate_mm_h"])
        if P.get("flow_rate_m3_h") and P.get("area_m2"):
            return float(P["flow_rate_m3_h"]) / float(P["area_m2"]) * 1000.0
        return None

    # ------------------------------------------------------------------
    def run(self, S: Series, state: dict, rain: np.ndarray, irr_gross: np.ndarray,
            et_mult=None, taw_mult=None, learn_k=1.0, i0=0, i1=None, callback=None, record=True, irr_infil=None):
        """
        Прогін балансу з годин i0..i1. rain/irr_gross — масиви (N, T) або (T,).
        irr_infil (1.10) — необов'язково, частка поливу, що вбирається, для кожної години (форма як irr_gross).
        Без нього діє стала частка Ksat / інтенсивність для дощування (1.2). Потрібно, щоб полив циклами
        з паузами (його вбирання розраховано в decision.infiltration_plan) моделювався узгоджено.
        callback(i, state, flux) — викликається після кожної години (для фільтра Калмана).
        Повертає (state, траєкторії {ключ: (N, T')}).

        Версія 1.2:
          * сніг: опади при T ≤ 0 °C накопичуються як сніговий запас (SWE) і тануть за градусо-денним
            методом SWAT+ (поріг 0.5 °C, коефіцієнт 0.5…4.5 мм/°C/добу із сезонним ходом); під снігом
            немає випаровування й транспірації (пропорційно частці покриття, повне покриття при SWE ≥ 25 мм);
          * мерзлий ґрунт (T ґрунту ≤ 0 °C на початку доби): CN береться для найвологішої умови (ARC III);
          * два шари кореневої зони: верхній 0–Z_UP (0.3 м) і нижній. Загальний дефіцит Dr рахується, як і
            раніше (від нього Ks і рішення), а дефіцит верхнього шару Du — окремо: опади й полив спершу
            заповнюють верхній шар, надлишок стікає вниз; випаровування — з верхнього; транспірація
            ділиться за розподілом коренів (правило 40-30-20-10) з поправкою на доступність води.
            Du потрібен, щоб датчик на 10–30 см порівнювався з вологістю СВОГО шару;
          * полив дощуванням з інтенсивністю вище Ksat частково стікає (консервативно: стала інфільтрація = Ksat).
        """
        P, crop, meth = self.P, self.crop, self.method
        T = len(S.times)
        i1 = T if i1 is None else i1
        N = len(state["dr"])
        rain = np.broadcast_to(np.atleast_2d(rain), (N, T)) if np.ndim(rain) == 1 else rain
        irr_gross = np.broadcast_to(np.atleast_2d(irr_gross), (N, T)) if np.ndim(irr_gross) == 1 else irr_gross
        if irr_infil is not None:
            irr_infil = np.broadcast_to(np.atleast_2d(np.asarray(irr_infil, float)), (N, T)) \
                if np.ndim(irr_infil) == 1 else np.asarray(irr_infil, float)
        et_mult = np.ones((N, T)) if et_mult is None else (np.broadcast_to(et_mult[:, None], (N, T))
                                                            if np.ndim(et_mult) == 1 else et_mult)
        taw_mult = np.ones(N) if taw_mult is None else np.asarray(taw_mult, float)
        if "du" not in state:
            state["du"] = np.clip(state["dr"], 0, None) * min(Z_UP, state["zr"]) / max(state["zr"], 1e-6)
        if "swe" not in state:
            state["swe"] = np.zeros(N)
        ea = meth["ea"] * float(P.get("ei_mult", 1.0))
        mulch_cov = float(P.get("mulch_cover", 0.0))
        e_mulch = 1.0 - 0.5 * mulch_cov if P.get("mulch_type") == "organic" else 1.0
        few_mulch = 0.15 if P.get("mulch_type") == "plastic" else 1.0
        tw_k = cropm.k_water_temperature(P.get("water_temp"), crop["season"])
        ec_iw = float(P.get("ec_iw", 0.3))
        hf = float(P.get("hydrophobicity", 0.0))
        lam = float(P.get("cn_lambda", 0.2))
        theta_fc = self.hyd["theta_fc"]
        rate = self.application_rate()
        infil_frac = 1.0
        if rate and meth["wind_sensitive"] and rate > self.hyd["ksat_mm_h"]:
            infil_frac = self.hyd["ksat_mm_h"] / rate
        ts_series = S.extra.get("ts")
        doy = S.extra.get("doy")
        keys = ("dr", "ks", "taw", "raw", "et", "e", "t", "dp", "ro", "pe", "inet", "cr", "ece", "theta", "de",
                "du", "theta_up", "swe", "melt")
        rec = {k: np.zeros((N, i1 - i0)) for k in keys} if record else None
        loss_frac = np.zeros(N)
        ro_frac = np.zeros(N)
        cur_day = None
        for i in range(i0, i1):
            t_air = float(S.t_air[i])
            # сезонний коефіцієнт танення SWAT+: (mx+mn)/2 + (mx−mn)/2·sin(2π/365·(doy−81))
            dd = float(doy[i]) if doy is not None else 180.0
            bmlt = (SNOW["meltmx"] + SNOW["meltmn"]) / 2 + (SNOW["meltmx"] - SNOW["meltmn"]) / 2 * \
                np.sin(2 * np.pi / 365.0 * (dd - 81.0))
            # --- поріг доби: стік (CN з динамічною умовою вологості) і перехоплення
            d = S.day_index[i]
            if d != cur_day:
                cur_day = d
                s0 = i
                e0 = s0
                while e0 < T and S.day_index[e0] == d:
                    e0 += 1
                tt = S.t_air[s0:e0]
                liq_mask = tt > SNOW["falltmp"]
                p_rain_day = (rain[:, s0:e0] * liq_mask[None, :]).sum(axis=1)
                p_snow_day = (rain[:, s0:e0] * (~liq_mask)[None, :]).sum(axis=1)
                melt_pot = bmlt / 24.0 * np.clip(tt - SNOW["melttmp"], 0, None).sum()
                melt_day = np.minimum(state["swe"] + p_snow_day, melt_pot)
                p_liq_day = p_rain_day + melt_day
                taw_now = self.taw(S.zr[i], taw_mult)
                wet = 1.0 - np.clip(state["dr"] / np.maximum(taw_now, 1e-6), 0, 1)
                frozen = ts_series is not None and np.isfinite(ts_series[i]) and ts_series[i] <= 0.0
                if frozen:
                    wet = np.ones(N)   # мерзлий ґрунт: умова найвищого стоку (ARC III)
                cn = soil.cn_effective(self.cn2 + hf * 10.0 * (1.0 - wet), wet)
                ro_d = soil.runoff_cn(p_liq_day, cn, lam)
                ic_d = soil.interception(p_rain_day, S.lai[i], S.fc[i])
                with np.errstate(divide="ignore", invalid="ignore"):
                    ro_frac = np.where(p_liq_day > 0, ro_d / p_liq_day, 0.0)
                    loss_frac = np.where(p_liq_day > 0, np.clip((ro_d + ic_d) / p_liq_day, 0, 1), 0.0)
            # --- ріст коренів: нова зона ґрунту входить зі своєю вологістю
            z = S.zr[i]
            if z > state["zr"]:
                add = 1000.0 * (theta_fc - self.theta_sub)
                state["dr"] = state["dr"] + add * (z - state["zr"])
                state["du"] = state["du"] + add * max(0.0, min(z, Z_UP) - min(state["zr"], Z_UP))
            state["zr"] = z
            zu = min(Z_UP, z)
            taw = self.taw(z, taw_mult)
            tawu = self.taw(zu, taw_mult)
            tawl = taw - tawu
            raw = S.p[i] * taw
            sat_cap = self.sat_unit * z
            wfc = 1000.0 * theta_fc * z
            # --- сніг
            pr = rain[:, i]
            snow = pr if t_air <= SNOW["falltmp"] else 0.0 * pr
            liq = pr - snow
            state["swe"] = state["swe"] + snow
            melt = np.minimum(state["swe"], bmlt / 24.0 * max(t_air - SNOW["melttmp"], 0.0))
            state["swe"] = state["swe"] - melt
            cover = np.clip(state["swe"] / SNOW["covmx"], 0.0, 1.0)
            # --- надходження води
            win = liq + melt
            pe = win * (1.0 - loss_frac)
            ro = win * ro_frac
            ig = irr_gross[:, i]
            inet_raw = ig * ea * (1.0 - S.drift[i])
            inet = inet_raw * (irr_infil[:, i] if irr_infil is not None else infil_frac)
            ro = ro + (inet_raw - inet)
            state["h_since_irr"] = np.where(ig > 0, 0.0, state["h_since_irr"] + 1.0)
            state["fw"] = np.where(ig > 0, meth["fw"], np.where(liq > 2.0, 1.0, state["fw"]))
            # --- випаровування з поверхні (FAO-56, подвійний Kc)
            few = np.clip(np.minimum(1.0 - S.fc[i], state["fw"]) * few_mulch, 0.01, 1.0)
            de = np.maximum(0.0, state["de"] - pe - inet / np.maximum(state["fw"], 0.05))
            kr = np.where(de > self.rew, np.clip((self.tew - de) / max(self.tew - self.rew, 1e-3), 0, 1), 1.0)
            ke = np.minimum(kr * (S.kcmax[i] - S.kcb[i]), few * S.kcmax[i])
            eto = S.eto[i] * et_mult[:, i]
            ev = np.maximum(0.0, ke * eto * e_mulch) * (1.0 - cover)
            state["de"] = np.minimum(self.tew, de + ev / few)
            # --- транспірація зі стресами
            ks_w = np.where(state["dr"] > raw, np.clip((taw - state["dr"]) / np.maximum((1 - S.p[i]) * taw, 1e-6), 0, 1), 1.0)
            ece = self.ece_base + state["x"]
            ks_s = salinity.ks_salinity(ece, crop["ece_thr"], crop["b"], crop["ky"])
            k_tw = np.where(state["h_since_irr"] < 24, tw_k, 1.0)
            lk = learn_k[0] if isinstance(learn_k, list) else learn_k   # список → змінюється фільтром «на льоту»
            tr = ks_w * ks_s * k_tw * S.kcb[i] * S.kx[i] * eto * lk * (1.0 - cover)
            # --- капілярне підживлення
            if self.cr_on:
                cr = soil.capillary_rise_day(P["gw_depth"], z, self.tex, state["dr"] / np.maximum(taw, 1e-6)) / 24.0
                cr = np.where(state["dr"] > 0, np.minimum(cr, tr), 0.0)
            else:
                cr = 0.0 * pe
            et = ev + tr
            # --- верхній шар (діагностичний підстан для датчика): спершу надходження, надлишок — униз
            du = state["du"] - pe - inet
            du = np.maximum(du, 0.0)
            du = du + ev
            two = np.asarray(tawl) > 1e-6                 # чи є нижній шар (корені глибші за Z_UP)
            r_up = root_share(zu / z)
            dl_now = state["dr"] - state["du"]
            a_u = np.clip(1.0 - du / np.maximum(tawu, 1e-6), 0.05, 1.0)
            a_l = np.clip(1.0 - dl_now / np.maximum(tawl, 1e-6), 0.05, 1.0)
            fu = np.where(two, r_up * a_u / np.maximum(r_up * a_u + (1.0 - r_up) * a_l, 1e-9), 1.0)
            du = np.minimum(du + tr * fu, tawu)
            dr = state["dr"] - pe - inet - cr + et
            # --- дренаж надлишку понад FC і насичення
            dp = np.where(dr < 0, -dr * self.tau_h, 0.0)
            dr = dr + dp
            sat_ex = np.where(dr < -sat_cap, -sat_cap - dr, 0.0)
            dr = np.minimum(np.maximum(dr, -sat_cap), taw)
            ro = ro + sat_ex
            du2 = np.maximum(du, dr - tawl)                # нижній шар не може бути сухішим за в'янення
            du2 = np.where(dr <= 0, 0.0, du2)              # уся зона на FC або вище → верхній шар теж на FC
            du = np.where(two, du2, np.maximum(dr, 0.0))
            state["dr"] = dr
            state["du"] = np.clip(du, 0.0, tawu)
            # --- солі
            state["x"] = salinity.leach(state["x"] + salinity.irrigation_delta_ece(inet, ec_iw, wfc)
                                        + S.fert_dece[i], dp, wfc)
            flux = dict(pe=pe, inet=inet, et=et, dp=dp, ro=ro, cr=cr, e=ev, t=tr, tu=tr * fu, taw=taw, raw=raw,
                        zu=zu, tawu=tawu, melt=melt)
            if callback is not None:
                callback(i, state, flux)
            if record:
                j = i - i0
                rec["dr"][:, j] = state["dr"]
                rec["ks"][:, j] = ks_w * ks_s
                rec["taw"][:, j] = taw
                rec["raw"][:, j] = raw
                rec["et"][:, j] = et
                rec["e"][:, j] = ev
                rec["t"][:, j] = tr
                rec["dp"][:, j] = dp
                rec["ro"][:, j] = ro
                rec["pe"][:, j] = pe
                rec["inet"][:, j] = inet
                rec["cr"][:, j] = cr
                rec["ece"][:, j] = self.ece_base + state["x"]
                rec["theta"][:, j] = theta_fc - state["dr"] / (1000.0 * z)
                rec["de"][:, j] = state["de"]
                rec["du"][:, j] = state["du"]
                rec["theta_up"][:, j] = theta_fc - state["du"] / (1000.0 * zu)
                rec["swe"][:, j] = state["swe"]
                rec["melt"][:, j] = melt
        return state, rec

    # ------------------------------------------------------------------
    def theta_from_dr(self, dr, zr):
        return self.hyd["theta_fc"] - np.asarray(dr) / (1000.0 * zr)

    def dr_from_theta(self, theta, zr):
        return 1000.0 * (self.hyd["theta_fc"] - np.asarray(theta)) * zr

    def summary(self):
        return dict(theta_fc=round(self.hyd["theta_fc"], 3), theta_wp=round(self.hyd["theta_wp"], 3),
                    theta_sat=round(self.hyd["theta_sat"], 3), ksat_mm_h=round(self.hyd["ksat_mm_h"], 1),
                    texture=self.tex, texture_uk=soil.TEXTURE_UK.get(self.tex, self.tex),
                    hsg=self.hsg, cn2=round(self.cn2, 1), tew=round(self.tew, 1), rew=round(self.rew, 1),
                    tau=round(self.tau, 3), awc_mm_per_m=round(self.taw_unit, 1),
                    ece_base=round(self.ece_base, 2))
