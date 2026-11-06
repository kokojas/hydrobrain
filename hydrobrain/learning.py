"""
Машинне навчання поверх фізичної моделі (feature engineering + онлайн-навчання).

UserPreferenceLearner — рекурсивний метод найменших квадратів (RLS) із забуванням: навчається після
КОЖНОЇ нової події, без перенавчання з нуля і без важких бібліотек.
   Ціль — відхилення користувача від БАЗОВОЇ норми моделі (до застосування навченого множника):
   система показала 25 мм при множнику ×1.25 → базова норма 20 мм; користувач полив 26 мм →
   y = ln(26/20). Наступного разу в схожому контексті базова норма множиться на exp(ŷ).
   Множник обмежено асиметрично (типово 0.6…1.25): система може навчитися поливати менше, але
   не може без обмежень навчитися переливати — надлишок понад потребу йде в дренаж.
   Окремо вчиться «поріг», коли користувач любить поливати (раніше/пізніше за модель) → множник MAD.

Адаптація моделі до конкретного поля (поправка Kc) робиться ЛИШЕ фільтром Калмана
(assimilation.py, змінна k). Раніше тут був ще ModelResidualLearner, який вчив ту саму поправку
з тих самих інновацій — його прибрано (див. CHANGELOG.md).

FeatureEngineer формує ознаки для учня і для журналу офлайн-навчання (CSV → sklearn).
"""
from __future__ import annotations

import csv
import math
import os
import threading
from datetime import datetime

import numpy as np


class FeatureEngineer:
    NAMES = ["bias", "dr_rel", "eto_3d", "vpd", "rain_3d", "rain_fc_24h", "rain_prob_24h", "dvs",
             "stage_sens", "ks", "ece_rel", "doy_sin", "doy_cos", "hour_sin", "hour_cos", "sand_rel",
             "days_since_irr"]

    def build(self, ctx: dict) -> np.ndarray:
        doy = ctx.get("doy", 180)
        hr = ctx.get("hour", 12)
        f = dict(
            bias=1.0,
            dr_rel=float(np.clip(ctx.get("dr_rel", 0.3), -0.5, 1.0)),
            eto_3d=ctx.get("eto_3d", 3.0) / 5.0,
            vpd=ctx.get("vpd", 1.0) / 2.0,
            rain_3d=math.log1p(ctx.get("rain_3d", 0.0)) / 3.0,
            rain_fc_24h=math.log1p(ctx.get("rain_fc_24h", 0.0)) / 3.0,
            rain_prob_24h=ctx.get("rain_prob_24h", 0.0) / 100.0,
            dvs=float(np.clip(ctx.get("dvs", 0.5), 0, 1.2)),
            stage_sens=1.0 if ctx.get("sensitive") else 0.0,
            ks=ctx.get("ks", 1.0),
            ece_rel=ctx.get("ece_rel", 0.0),
            doy_sin=math.sin(2 * math.pi * doy / 365.0),
            doy_cos=math.cos(2 * math.pi * doy / 365.0),
            hour_sin=math.sin(2 * math.pi * hr / 24.0),
            hour_cos=math.cos(2 * math.pi * hr / 24.0),
            sand_rel=ctx.get("sand", 30.0) / 100.0,
            days_since_irr=min(ctx.get("days_since_irr", 10.0), 30.0) / 10.0,
        )
        return np.array([f[n] for n in self.NAMES], dtype=float)

    def as_dict(self, x):
        return dict(zip(self.NAMES, map(float, x)))


class RLS:
    """Рекурсивні найменші квадрати з коефіцієнтом забування lam і L2-регуляризацією через P0."""

    def __init__(self, n, lam=0.98, delta=10.0):
        self.w = np.zeros(n)
        self.P = np.eye(n) * delta
        self.lam = lam
        self.n_updates = 0

    def predict(self, x):
        return float(self.w @ x)

    def update(self, x, y):
        Px = self.P @ x
        k = Px / (self.lam + x @ Px)
        e = y - self.w @ x
        self.w = self.w + k * e
        self.P = (self.P - np.outer(k, Px)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)
        # запобіжник від розбіжності
        if not np.all(np.isfinite(self.P)) or np.trace(self.P) > 1e6:
            self.P = np.eye(len(self.w)) * 10.0
        self.n_updates += 1
        return float(e)

    def to_dict(self):
        return dict(w=self.w.tolist(), P=self.P.tolist(), lam=self.lam, n=self.n_updates)

    @classmethod
    def from_dict(cls, d):
        r = cls(len(d["w"]), d.get("lam", 0.98))
        r.w = np.array(d["w"])
        r.P = np.array(d["P"])
        r.n_updates = d.get("n", 0)
        return r


class UserPreferenceLearner:
    """Навчається на тому, наскільки користувач змінює рекомендовану норму і час поливу."""

    def __init__(self, fe: FeatureEngineer, shrink_k=3.0, max_up=1.25, max_down=0.6):
        self.fe = fe
        self.rls = RLS(len(fe.NAMES), lam=0.97, delta=1.0)
        self.k = shrink_k
        self.log_up = math.log(max_up)
        self.log_down = math.log(max_down)
        self.mad_factor = 1.0
        self.n_timing = 0

    @property
    def max_up(self) -> float:
        return float(math.exp(self.log_up))

    def multiplier(self, ctx) -> float:
        n = self.rls.n_updates
        if n == 0:
            return 1.0
        y = self.rls.predict(self.fe.build(ctx)) * n / (n + self.k)   # стягування до 1 на малій кількості правок
        return float(math.exp(float(np.clip(y, self.log_down, self.log_up))))

    def observe_amount(self, ctx, base_mm, applied_mm):
        """
        base_mm — норма моделі БЕЗ навченого множника (показана норма ÷ множник, що діяв тоді).
        Ціль y = ln(applied / base): учень бачить повне відхилення користувача від моделі, а не
        залишок відносно вже скоригованої норми (інакше поправка відставала б або коливалась).
        """
        if base_mm is None or base_mm <= 0.5 or applied_mm is None or applied_mm < 0:
            return None
        y = float(np.clip(math.log(max(applied_mm, 0.1) / base_mm), math.log(0.5), math.log(2.0)))
        return self.rls.update(self.fe.build(ctx), y)

    def observe_timing(self, dr_at_irrigation, raw):
        """Користувач полив сам, коли дефіцит був dr; модель вважала порогом raw → згладжуємо MAD."""
        if raw <= 1.0:
            return
        r = float(np.clip(dr_at_irrigation / raw, 0.4, 1.5))
        a = 1.0 / (self.n_timing + 4.0)
        self.mad_factor = float(np.clip((1 - a) * self.mad_factor + a * r, 0.6, 1.2))
        self.n_timing += 1

    def to_dict(self):
        return dict(rls=self.rls.to_dict(), mad_factor=self.mad_factor, n_timing=self.n_timing)

    def load(self, d):
        if d:
            self.rls = RLS.from_dict(d["rls"])
            self.mad_factor = d.get("mad_factor", 1.0)
            self.n_timing = d.get("n_timing", 0)


_TRAINING_LOG_LOCK = threading.Lock()


def append_training_row(path, features: dict, targets: dict, meta: dict):
    # Different fields share one CSV; per-field locks cannot protect its header.
    with _TRAINING_LOG_LOCK:
        _append_training_row(path, features, targets, meta)


def _append_training_row(path, features: dict, targets: dict, meta: dict):
    """Журнал для офлайн-навчання важчих моделей (градієнтний бустинг тощо)."""
    row = {**meta, **{f"f_{k}": v for k, v in features.items()}, **{f"y_{k}": v for k, v in targets.items()}}
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def offline_fit(csv_path, target="y_amount_logratio"):
    """
    Приклад офлайн-навчання на накопиченому журналі (потрібен scikit-learn).
    Повертає модель і важливість ознак — корисно для аналізу, які фактори впливають на правки.
    """
    try:
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as e:
        raise RuntimeError("потрібні pandas і scikit-learn") from e
    df = pd.read_csv(csv_path)
    df = df[df[target].notna()]
    X = df[[c for c in df.columns if c.startswith("f_")]]
    y = df[target]
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8)
    model.fit(X, y)
    imp = sorted(zip(X.columns, model.feature_importances_), key=lambda t: -t[1])
    return model, imp
