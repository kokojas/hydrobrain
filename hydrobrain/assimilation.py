"""
Злиття «модель + датчик» (розширений фільтр Калмана, EKF), щоб оцінити СПРАВЖНЮ вологість,
а не лише те, що показує датчик, і паралельно «підлаштувати» модель під конкретне поле.

Стан (v1.2) x = [Dr, b, k, Du]:
  Dr — дефіцит вологи всієї кореневої зони, мм (0 = польова вологоємність);
  b  — систематичне зміщення датчика, м³/м³ (калібрування);
  k  — поправка транспірації поля (множник до Kcb), безрозмірний, ≈ 1;
  Du — дефіцит верхнього шару 0–0.3 м, мм (новий у 1.2; у ≤ 1.1 його не було).

Прогноз (модель водного балансу за годину):  Dr ← Dr_model(k), Du ← Du_model(k);
    ∂Dr/∂k = T/k,  ∂Du/∂k = T_up/k  (T_up — частка транспірації з верхнього шару).
Спостереження датчика залежить від шару, де він стоїть:
    верхній шар:  z = θFC − Du/(1000·Zu) + b + v
    нижній шар:   z = θFC − (Dr − Du)/(1000·(Zr − Zu)) + b + v
    один шар (корені ще ≤ Zu):  z = θFC − Dr/(1000·Zr) + b + v
    v ~ N(0, σ_sens² + σ_spat² [+ 0.02², якщо датчик мілкіший за 5 см])

Чому Du — окрема змінна фільтра. Після дощу верхній шар на FC, а нижній сухий; датчик на 20 см тоді нічого
не каже про нижній шар. Якщо Du — лише похідна від Dr (як в першому варіанті 1.2), будь-яка помилка Du
«звалюється» на зміщення датчика b. З Du у стані фільтр виправляє саме той шар, який бачить датчик.

Чому b і k. Постійна різниця «датчик ↔ модель» пояснюється зміщенням b, різниця, що НАРОСТАЄ між
вимірами, — неточним Kc (k).

Якір FC (з 1.1, змінено в 1.2): коли модель вважає ґрунт насиченим і вода стікає, подається
псевдо-спостереження «Dr ≈ 0, Du ≈ 0» з похибкою 4% TAW. У 1.2 — ЛИШЕ якщо протягом 24 год датчик
незалежно підтвердив зволоження (стрибок θ ≥ 0.02 або θ біля FC). Без підтвердження модель могла помилитися
з дощем (злива пройшла повз поле), і якір дав би фальшиву впевненість.

Невизначеність моделі Q: похибка ET переважно систематична (σ росте як e·ΣET), плюс похибки опадів,
поливу і дренажу. Похибки Dr і Du корельовані (та сама вода), ρ = 0.7.
Виміри з інновацією > 4σ відкидаються (підозра на збій датчика).
"""
from __future__ import annotations

import math

import numpy as np

I_DR, I_B, I_K, I_DU = 0, 1, 2, 3
RHO_DU = 0.7


class SoilMoistureKalman:
    def __init__(self, dr0: float, sigma_dr0: float, sigma_bias0: float = 0.02, sigma_k0: float = 0.15,
                 k0: float = 1.0, q_bias_per_day: float = 0.0003, q_k_per_day: float = 0.02,
                 rel_err_et=0.10, rel_err_rain=0.25, rel_err_irr=0.10, pin_at_fc=True, pin_requires_sensor=True,
                 du0: float | None = None, sigma_du0: float | None = None):
        du0 = max(float(dr0), 0.0) * 0.3 if du0 is None else float(du0)
        sdu = sigma_dr0 * 0.5 if sigma_du0 is None else float(sigma_du0)
        self.x = np.array([float(dr0), 0.0, float(k0), du0])
        self.Pm = np.diag([sigma_dr0 ** 2, sigma_bias0 ** 2, sigma_k0 ** 2, sdu ** 2])
        self.Pm[I_DR, I_DU] = self.Pm[I_DU, I_DR] = RHO_DU * sigma_dr0 * sdu
        self.q_b = (q_bias_per_day ** 2) / 24.0
        self.q_k = (q_k_per_day ** 2) / 24.0
        self.e_et, self.e_rain, self.e_irr = rel_err_et, rel_err_rain, rel_err_irr
        self.et_cum = 0.0
        self.etu_cum = 0.0
        self.n_obs = 0
        self.pin = bool(pin_at_fc)
        self.n_pins = 0
        self.pin_requires_sensor = bool(pin_requires_sensor)
        self.h = 0
        self.pin_ok_until = -1
        self.last_obs_theta = None
        self.last_obs_h = None
        # моменти, коли МОДЕЛЬ вважала шар зволоженим до FC (для якоря з підтвердженням датчиком пізніше)
        self.wet_u_h = None
        self.wet_dr_h = None
        self.etu_since_wet = 0.0
        self.et_since_wet = 0.0
        self._taw = 0.0
        self._tawu = 0.0
        self.log = []

    # ------------------------------------------------------------------
    def predict(self, dr_model: float, flux: dict, du_model: float | None = None):
        """Після кроку моделі: x[Dr], x[Du] = прогноз моделі; P = F P Fᵀ + Q."""
        self.x[I_DR] = float(dr_model)
        if du_model is not None:
            self.x[I_DU] = float(du_model)
        self.h += 1
        m = lambda key: float(np.mean(flux.get(key, 0.0)))
        et_h, t_h, tu_h, e_h = m("et"), m("t"), m("tu"), m("e")
        etu_h = e_h + tu_h
        F = np.eye(4)
        F[I_DR, I_K] = t_h / max(self.x[I_K], 0.3)
        F[I_DU, I_K] = tu_h / max(self.x[I_K], 0.3)
        q_sys = self.e_et ** 2 * (2.0 * self.et_cum * et_h + et_h ** 2)
        qu_sys = self.e_et ** 2 * (2.0 * self.etu_cum * etu_h + etu_h ** 2)
        self.et_cum += et_h
        self.etu_cum += etu_h
        q_in = (self.e_rain * m("pe")) ** 2 + (self.e_irr * m("inet")) ** 2
        q = q_sys + q_in + (0.3 * m("dp")) ** 2 + 0.02 ** 2
        qu = qu_sys + q_in + 0.02 ** 2
        Q = np.diag([q, self.q_b, self.q_k, qu])
        Q[I_DR, I_DU] = Q[I_DU, I_DR] = RHO_DU * math.sqrt(q * qu)
        self.Pm = F @ self.Pm @ F.T + Q
        taw = m("taw")
        tawu = m("tawu") or taw
        self._taw, self._tawu = taw, tawu
        dp = m("dp")
        self.etu_since_wet += etu_h
        self.et_since_wet += et_h
        du_now = float(self.x[I_DU])
        if du_now <= 0.5 and m("pe") + m("inet") >= 1.0:      # модель: верхній шар щойно заповнено до FC
            self.wet_u_h, self.etu_since_wet = self.h, 0.0
        if dr_model <= 0.5 and dp >= 0.05:                     # модель: уся зона на FC, іде дренаж
            self.wet_dr_h, self.et_since_wet = self.h, 0.0
        confirmed = (not self.pin_requires_sensor) or self.h <= self.pin_ok_until
        if self.pin and confirmed and taw > 0 and dr_model <= 0.5 and dp >= 0.05:
            self._pin(I_DR, 0.04 * taw)
            self._pin(I_DU, 0.04 * tawu)
            self.n_pins += 1
        for idx, cap_taw in ((I_DR, taw), (I_DU, tawu)):
            if cap_taw > 0:   # дефіцит фізично обмежений [0, TAW] → невизначеність не росте безмежно
                cap = (0.6 * cap_taw) ** 2
                if self.Pm[idx, idx] > cap:
                    s = math.sqrt(cap / self.Pm[idx, idx])
                    self.Pm[idx, :] *= s
                    self.Pm[:, idx] *= s

    # ------------------------------------------------------------------
    def _pin(self, idx: int, sigma_pin_mm: float):
        """
        Псевдо-спостереження «дефіцит ≈ значення моделі» з похибкою σ_pin. Ефективну похибку R підібрано
        так, щоб апостеріорна дисперсія дорівнювала σ_pin² (а не зменшувалась з кожною годиною дренажу).
        """
        p00 = float(self.Pm[idx, idx])
        s2 = sigma_pin_mm ** 2
        if p00 <= s2 * 1.05:
            return
        R = s2 * p00 / (p00 - s2)
        H = np.zeros(4)
        H[idx] = 1.0
        K = self.Pm @ H / (p00 + R)
        I_KH = np.eye(4) - np.outer(K, H)
        self.Pm = I_KH @ self.Pm @ I_KH.T + np.outer(K, K) * R
        self.Pm = 0.5 * (self.Pm + self.Pm.T)

    # ------------------------------------------------------------------
    def update(self, theta_obs: float, zr: float, theta_fc: float, sigma_sens=0.02, sigma_spat=0.02,
               depth_cm=None, dr_bounds=(-1e9, 1e9), t=None, layer=None):
        """
        Корекція за показом датчика θ (м³/м³ або %). Повертає словник з інновацією та прапорцями.
        layer — dict(name="upper"|"lower"|"single", zu=товщина верхнього шару м, tawu=TAW верхнього шару мм).
        Без layer — модель «одного резервуара» (як у ≤ 1.1).
        """
        if theta_obs > 1.0:
            theta_obs = theta_obs / 100.0
        sig_depth = 0.0
        if layer is not None:
            if depth_cm is not None and depth_cm < 5:
                sig_depth = 0.02
        elif depth_cm is not None:
            zc = zr * 100.0
            if depth_cm < 0.15 * zc or depth_cm > 0.85 * zc:
                sig_depth = 0.02
        R = sigma_sens ** 2 + sigma_spat ** 2 + sig_depth ** 2
        if self.n_obs == 0:
            # перший показ: історія моделі без вимірів могла «забрести» → розширюємо невизначеність дефіцитів,
            # обнуляємо їх накопичені кореляції з b і k (кореляцію Dr–Du лишаємо)
            taw_guess = 1000.0 * zr * 0.25
            for idx, sc in ((I_DR, 0.5 * taw_guess), (I_DU, 0.5 * taw_guess * min(1.0, 0.3 / max(zr, 1e-6)))):
                self.Pm[idx, idx] = max(self.Pm[idx, idx], sc ** 2)
                for j in (I_B, I_K):
                    self.Pm[idx, j] = self.Pm[j, idx] = 0.0
            # незаписані дощі/поливи змінюють обидва шари разом → після розширення відновлюємо кореляцію ρ
            self.Pm[I_DR, I_DU] = self.Pm[I_DU, I_DR] = RHO_DU * math.sqrt(self.Pm[I_DR, I_DR] * self.Pm[I_DU, I_DU])
        # якір FC «заднім числом»: датчик зараз підтверджує зволоження, а модель вважала шар на FC
        # у попередні 48 год → спершу фіксуємо дефіцит шару (σ = 4% TAW + 10% ET після зволоження),
        # і лише потім застосовуємо показ — тоді різниця «датчик − модель» іде у зміщення b
        if self.pin and self.n_obs > 0:
            jump = (self.last_obs_theta is not None and self.last_obs_h is not None
                    and self.h - self.last_obs_h <= 48 and theta_obs - self.last_obs_theta >= 0.02)
            near = theta_obs - self.x[I_B] >= theta_fc - 0.02
            if jump or near:
                pinned = False
                if self.wet_u_h is not None and 0 <= self.h - self.wet_u_h <= 48 and self._tawu > 0:
                    self._pin(I_DU, 0.04 * self._tawu + 0.1 * self.etu_since_wet)
                    self.wet_u_h, pinned = None, True
                if self.wet_dr_h is not None and 0 <= self.h - self.wet_dr_h <= 48 and self._taw > 0:
                    self._pin(I_DR, 0.04 * self._taw + 0.1 * self.et_since_wet)
                    self.wet_dr_h, pinned = None, True
                if pinned:
                    self.n_pins += 1
        H = np.zeros(4)
        H[I_B] = 1.0
        name = "single" if layer is None else layer.get("name", "single")
        if name == "upper":
            zu = float(layer["zu"])
            H[I_DU] = -1.0 / (1000.0 * zu)
            z_pred = theta_fc - self.x[I_DU] / (1000.0 * zu) + self.x[I_B]
        elif name == "lower":
            zu = float(layer["zu"])
            zl = max(zr - zu, 0.05)
            H[I_DR] = -1.0 / (1000.0 * zl)
            H[I_DU] = 1.0 / (1000.0 * zl)
            z_pred = theta_fc - (self.x[I_DR] - self.x[I_DU]) / (1000.0 * zl) + self.x[I_B]
        else:
            H[I_DR] = -1.0 / (1000.0 * zr)
            z_pred = theta_fc - self.x[I_DR] / (1000.0 * zr) + self.x[I_B]
        nu = theta_obs - z_pred
        S = float(H @ self.Pm @ H + R)
        ok = abs(nu) <= 4.0 * math.sqrt(S)
        info = dict(time=t.isoformat() if t is not None else None, theta_obs=theta_obs,
                    theta_pred=float(z_pred), innovation=float(nu), sigma=math.sqrt(S), accepted=bool(ok),
                    dr_prior=float(self.x[I_DR]), du_prior=float(self.x[I_DU]), layer=name)
        if ok:
            self.n_obs += 1
            self.et_cum = 0.0
            self.etu_cum = 0.0
            K = self.Pm @ H / S
            self.x = self.x + K * nu
            self.x[I_DR] = float(np.clip(self.x[I_DR], *dr_bounds))
            tawu = float(layer.get("tawu", 1e9)) if layer is not None else 1e9
            self.x[I_DU] = float(np.clip(self.x[I_DU], 0.0, min(tawu, max(dr_bounds[1], 0.0))))
            if name == "single":
                self.x[I_DU] = max(self.x[I_DR], 0.0)
            self.x[I_B] = float(np.clip(self.x[I_B], -0.15, 0.15))
            self.x[I_K] = float(np.clip(self.x[I_K], 0.6, 1.5))
            I_KH = np.eye(4) - np.outer(K, H)
            self.Pm = I_KH @ self.Pm @ I_KH.T + np.outer(K, K) * R   # форма Джозефа (стійка)
            self.Pm = 0.5 * (self.Pm + self.Pm.T)
            wet_jump = (self.last_obs_theta is not None and self.last_obs_h is not None
                        and self.h - self.last_obs_h <= 48 and theta_obs - self.last_obs_theta >= 0.02)
            near_fc = theta_obs - self.x[I_B] >= theta_fc - 0.02
            if wet_jump or near_fc:
                self.pin_ok_until = self.h + 24
            self.last_obs_theta, self.last_obs_h = theta_obs, self.h
        info["dr_post"] = float(self.x[I_DR])
        info["du_post"] = float(self.x[I_DU])
        info["bias"] = float(self.x[I_B])
        info["k_field"] = float(self.x[I_K])
        info["innovation_mm"] = float(info["dr_post"] - info["dr_prior"])
        self.log.append(info)
        return info

    # ------------------------------------------------------------------
    def update_dr(self, dr_obs: float, sigma_mm: float, dr_bounds=(-1e9, 1e9), source="pseudo", t=None):
        """
        «Слабкий вимір» дефіциту кореневої зони напряму (без свіжого датчика, 1.7–1.8): оцінка з реаналізу
        ERA5-Land або прокручений старий показ датчика. Оновлюються лише Dr і Du (через їхню кореляцію);
        зміщення датчика і поправка Kc не чіпаються — інакше старий показ датчика врахувався б двічі.
        """
        R = float(sigma_mm) ** 2
        n = len(self.x)
        p00 = float(self.Pm[I_DR, I_DR])
        S = p00 + R
        K = np.zeros(n)
        K[I_DR] = p00 / S
        if n > I_DU:
            K[I_DU] = float(self.Pm[I_DU, I_DR]) / S
        nu = float(dr_obs) - float(self.x[I_DR])
        self.x = self.x + K * nu
        self.x[I_DR] = float(np.clip(self.x[I_DR], *dr_bounds))
        if n > I_DU:
            self.x[I_DU] = float(max(self.x[I_DU], -1e9))
        H = np.zeros(n)
        H[I_DR] = 1.0
        I_KH = np.eye(n) - np.outer(K, H)
        self.Pm = I_KH @ self.Pm @ I_KH.T + np.outer(K, K) * R
        self.Pm = 0.5 * (self.Pm + self.Pm.T)
        info = dict(time=t.isoformat() if t is not None else None, source=source, dr_obs=float(dr_obs),
                    sigma_mm=float(sigma_mm), gain=float(K[I_DR]), dr_post=float(self.x[I_DR]))
        self.log.append(info)
        return info

    @property
    def dr(self):
        return float(self.x[I_DR])

    @property
    def du(self):
        return float(self.x[I_DU])

    @property
    def sigma_dr(self):
        return float(math.sqrt(max(self.Pm[I_DR, I_DR], 0.0)))

    @property
    def sigma_du(self):
        return float(math.sqrt(max(self.Pm[I_DU, I_DU], 0.0)))

    @property
    def bias(self):
        return float(self.x[I_B])

    @property
    def sigma_bias(self):
        return float(math.sqrt(max(self.Pm[I_B, I_B], 0.0)))

    @property
    def k_field(self):
        return float(self.x[I_K])

    def to_dict(self):
        return dict(x=self.x.tolist(), P=self.Pm.tolist(), et_cum=self.et_cum, etu_cum=self.etu_cum,
                    n_obs=self.n_obs, pin=self.pin, n_pins=self.n_pins, pin_requires_sensor=self.pin_requires_sensor,
                    h=self.h, pin_ok_until=self.pin_ok_until, last_obs_theta=self.last_obs_theta,
                    last_obs_h=self.last_obs_h, wet_u_h=self.wet_u_h, wet_dr_h=self.wet_dr_h,
                    etu_since_wet=self.etu_since_wet, et_since_wet=self.et_since_wet,
                    taw=self._taw, tawu=self._tawu)

    @classmethod
    def from_dict(cls, d, du_default=None):
        k = cls(0.0, 1.0)
        x = list(d["x"])
        P = np.array(d["P"], dtype=float)
        n = len(x)
        if n < 4:   # сумісність зі станами 1.0 (2 змінні) і 1.1 (3 змінні)
            x = x + [1.0] * max(0, 3 - n)
            x.append(max(x[0], 0.0) * 0.3 if du_default is None else float(du_default))
            P2 = np.diag([1.0, 0.02 ** 2, 0.15 ** 2, 0.0])
            P2[:n, :n] = P
            sd = math.sqrt(max(P2[0, 0], 1.0)) * 0.5
            P2[3, 3] = sd ** 2
            P2[0, 3] = P2[3, 0] = RHO_DU * math.sqrt(P2[0, 0]) * sd
            P = P2
        k.x = np.array(x, dtype=float)
        k.Pm = P
        k.et_cum = float(d.get("et_cum", 0.0))
        k.etu_cum = float(d.get("etu_cum", 0.0))
        k.n_obs = int(d.get("n_obs", 1))
        k.pin = bool(d.get("pin", True))
        k.n_pins = int(d.get("n_pins", 0))
        k.pin_requires_sensor = bool(d.get("pin_requires_sensor", True))
        k.h = int(d.get("h", 0))
        k.pin_ok_until = int(d.get("pin_ok_until", -1))
        k.last_obs_theta = d.get("last_obs_theta")
        k.last_obs_h = d.get("last_obs_h")
        k.wet_u_h = d.get("wet_u_h")
        k.wet_dr_h = d.get("wet_dr_h")
        k.etu_since_wet = float(d.get("etu_since_wet", 0.0))
        k.et_since_wet = float(d.get("et_since_wet", 0.0))
        k._taw = float(d.get("taw", 0.0))
        k._tawu = float(d.get("tawu", 0.0))
        return k
