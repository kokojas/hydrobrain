"""
Короткі людські пояснення рекомендації українською.

summary — 1-2 речення «що робити і чому» (головне);
details — список причин із числами (для розгорнутого вигляду в інтерфейсі).
"""
from __future__ import annotations

ACTION_UK = {"irrigate_now": "Поливати зараз", "postpone": "Відкласти полив", "skip": "Не поливати",
             "measure": "Потрібен вимір вологості"}
# орієнтовні формулювання для поля без датчика (1.8): дія та сама, але без хибної точності
ACTION_UK_ORIENT = {"irrigate_now": "Ймовірно, потрібен полив", "postpone": "Ймовірно, полив знадобиться згодом",
                    "skip": "Ймовірно, полив поки не потрібен"}


def _h(hours):
    if hours is None:
        return "—"
    hours = float(hours)
    if hours < 1.5:
        return "~1 год"
    if hours < 36:
        return f"~{hours:.0f} год"
    return f"~{hours / 24:.1f} доби"


def _pct(p):
    return f"{100 * p:.0f}%"


def reason_text(r):
    c = r["code"]
    if c == "off_season":
        return "Культура поза вегетацією (до сходів або після збирання) — полив не потрібен."
    if c == "critical_deficit":
        return (f"Критичний дефіцит: бракує {r['dr']:.0f} мм із {r['taw']:.0f} мм доступної води, "
                f"рослина вже сповільнює випаровування (Ks={r['ks']:.2f}).")
    if c == "stress_started":
        return (f"Легкодоступну воду вичерпано: дефіцит {r['dr']:.0f} мм ≥ порогу {r['raw']:.0f} мм "
                f"(Ks={r['ks']:.2f}).")
    if c == "rain_rescue":
        return (f"Протягом {_h(r['hours'])} очікується ~{r['mm']:.0f} мм дощу, що з імовірністю "
                f"{_pct(r['p'])} покриє дефіцит.")
    if c == "rain_decisive":
        when = f" через {_h(r['hours'])}" if r.get("hours") is not None else ""
        return (f"Саме дощ знімає потребу: очікується ~{r['mm']:.0f} мм за 48 год{when}; без нього ймовірність "
                f"стресу була б {_pct(r['p_dry'])}.")
    if c == "rain_expected":
        when = f", перший — через {_h(r['hours'])}" if r.get("hours") is not None else ""
        return f"За прогнозом ~{r['mm']:.0f} мм опадів (макс. імовірність {_pct(r['p'])}{when})."
    if c == "enough_water":
        tail = (f"поріг буде досягнуто приблизно через {_h(r['hours'])}." if r.get("hours") is not None
                else "за типовим (медіанним) сценарієм поріг у межах горизонту прогнозу не буде досягнуто.")
        return f"Запасу вологи достатньо: дефіцит {r['dr']:.0f} мм < порогу {r['raw']:.0f} мм, " + tail
    if c == "will_need_soon":
        return (f"Поріг стресу ({r['raw']:.0f} мм) буде перетнуто приблизно через {_h(r['hours'])} "
                f"(імовірність {_pct(r['p'])} за {r.get('horizon_h', 48)} год).")
    if c == "better_window":
        return f"Краще вікно для поливу — через {_h(r['hours'])} (менше випаровування/вітру)."
    if c == "wind_high":
        return f"Сильний вітер {r['wind']:.1f} м/с зносить краплі — полив перенесено на {_h(r['hours'])}."
    if c == "heat":
        return f"Спека до {r['tmax']:.0f}°C — поливати рано вранці або ввечері."
    if c == "salinity":
        return (f"Засолення ECe {r['ece']:.1f} дСм/м близьке до порогу культури {r['thr']:.1f} — "
                f"додано норму промивання {100 * r['lr']:.0f}%.")
    if c == "sensitive_stage":
        return "Культура в чутливій до нестачі води фазі — поріг поливу знижено."
    if c == "reduced_for_rain":
        return f"Норму зменшено (×{r['factor']:.2f}), щоб не втрачати воду на дренаж після дощу."
    if c == "watch":
        return (f"Ситуація прикордонна (імовірність стресу {_pct(r['p'])} за {r.get('horizon_h', 48)} год) — "
                f"перевірити знову через {_h(r['hours'])}.")
    if c == "rain_wait_limit":
        return (f"Дощ уже обіцяли {r['n']} рази поспіль, а стрес триває — більше не чекаємо, поливаємо.")
    if c == "recheck_after_rain":
        return f"Перевірити знову через {_h(r['hours'])}, після дощу."
    if c == "rain_insufficient":
        when = f" через {_h(r['hours'])}" if r.get("hours") is not None else ""
        return f"Очікуваних ~{r['mm']:.0f} мм дощу{when} недостатньо, щоб зняти дефіцит — їх враховано в нормі."
    if c == "wind_losses":
        return f"Вітер {r['wind']:.1f} м/с: втрати на знос крапель ~{100 * r['loss']:.0f}% — враховано в нормі."
    if c == "split":
        return (f"Дефіцит великий ({r['total']:.0f} мм нетто) — за один раз дається {r['now']:.0f} мм, "
                f"решту — наступними поливами (щоб вода не пішла в стік/дренаж).")
    if c == "learned_adjust":
        return f"Норму скориговано на ваші попередні правки (×{r['mult']:.2f})."
    if c == "user_overwater":
        return (f"Ви зазвичай поливаєте більше, ніж потрібно за моделлю; норму збільшено лише до ×{r['mult']:.2f}, "
                f"бо надлишок понад потребу йде в дренаж. Якщо вважаєте, що модель недооцінює потребу, "
                f"перевірте датчик або параметри ґрунту.")
    if c == "sensor_fault":
        return "Показ датчика сильно відрізняється від моделі — можливий збій датчика, перевірте його."
    if c == "infiltration_cycles":
        return (f"Інтенсивність дощування {r['rate_mm_h']:.0f} мм/год перевищує швидкість вбирання ґрунту "
                f"({r['ksat_mm_h']:.1f} мм/год): безперервно вода стікала б з поля. Поливайте {r['cycles']} циклами "
                f"до ≈ {r['on_min']} хв (останній ≈ {r.get('last_on_min', r['on_min'])} хв) "
                f"з паузами ≈ {r['soak_min']} хв (разом ≈ {r['total_h']:.1f} год) — "
                f"так уся вода вбереться.")
    if c == "infiltration_limited":
        return (f"Ґрунт вбирає воду повільно ({r['ksat']:.1f} мм/год при інтенсивності {r['rate']:.0f} мм/год): за добу "
                f"навіть {r['cycles']} циклами з паузами вбереться лише ≈ {r['now']:.0f} мм із потрібних "
                f"{r['total']:.0f} мм. Решту — наступного поливу; або зменшіть інтенсивність дощування.")
    if c == "reduced_for_drainage":
        return (f"Повна норма частково пішла б глибше за корені (очікувані втрати ≈ {r['waste']:.0f} мм) — "
                f"норму зменшено до {r['factor'] * 100:.0f}%.")
    if c == "state_reanalysis":
        lag = f" (дані ≈ {r['lag']:.0f} діб тому)" if r.get("lag") is not None else ""
        return ("Датчика немає: стан ґрунту оцінено з реаналізу ERA5-Land у цій точці" + lag + " і прокручено до "
                "сьогодні ML-моделлю.")
    if c == "state_stale_sensor":
        return (f"Останній показ датчика був ≈ {r['lag']:.0f} діб тому — стан прокручено до сьогодні ML-моделлю "
                f"(що старіший показ, то нижча точність).")
    if c == "orientative_no_sensor":
        return ("Це орієнтовна рекомендація без датчика: на перевірці така оцінка правильно визначала «сухо/волого» "
                "приблизно в 3 випадках із 4 і пропускала близько третини сухих днів. Перед поливом перевірте "
                "вологість ґрунту (датчик або ручна проба); навіть один вимір на тиждень помітно підвищує точність.")
    if c == "measurement_required":
        return "Рекомендації без вимірювання вимкнено — додайте показ датчика або ручний вимір вологості."
    if c == "no_sensor":
        return "Немає свіжих показів датчика — вологість оцінена лише моделлю (нижча впевненість)."
    return c


PRIORITY = ["critical_deficit", "stress_started", "split", "rain_wait_limit", "rain_rescue", "rain_insufficient", "recheck_after_rain", "rain_decisive", "off_season", "will_need_soon",
            "enough_water", "watch", "rain_expected", "wind_high", "better_window", "salinity", "sensitive_stage",
            "heat", "wind_losses", "reduced_for_rain", "reduced_for_drainage", "infiltration_cycles", "infiltration_limited", "learned_adjust", "user_overwater", "state_reanalysis", "state_stale_sensor", "orientative_no_sensor", "sensor_fault", "no_sensor"]


def _cycles_suffix(decision):
    """1.10: якщо поливати треба циклами (інтенсивність дощування > Ksat), це видно вже в заголовку."""
    cy = ((decision.get("amount") or {}).get("cycles") or {})
    return f", {cy['cycles']} циклами з паузами" if cy.get("cycles", 0) > 1 else ""


def build(decision: dict, moisture: dict, local_time_str: str | None, orientative: bool = False):
    reasons = sorted(decision["reasons"], key=lambda r: PRIORITY.index(r["code"]) if r["code"] in PRIORITY else 99)
    act = decision["action"]
    head = ACTION_UK[act]
    am = decision.get("amount")
    if orientative:
        # без датчика: та сама дія, але діапазон норми замість точної цифри; пояснення про точність — першим
        head = ACTION_UK_ORIENT.get(act, head)
        rng = (am or {}).get("range_mm")
        if act == "irrigate_now" and rng:
            head += f": орієнтовно {rng[0]:.0f}–{rng[1]:.0f} мм"
        elif act == "postpone":
            if local_time_str:
                head += f" (≈ {local_time_str})"
            if rng:
                head += f", орієнтовно {rng[0]:.0f}–{rng[1]:.0f} мм"
            elif decision.get("waiting_for_rain"):
                head += " — очікується дощ"
        orient = [r for r in reasons if r["code"] == "orientative_no_sensor"]
        rest = [r for r in reasons if r["code"] != "orientative_no_sensor"]
        main = [reason_text(r) for r in rest[:1] + orient]
        summary = head + ". " + " ".join(main) + f" Впевненість {decision['confidence_label']}."
        return dict(headline=head + _cycles_suffix(decision), summary=summary, details=[reason_text(r) for r in orient + rest])
    if act == "irrigate_now" and am:
        head += f": {am['gross_mm']:.0f} мм ({am['liters_per_m2']:.0f} л/м²)"
    elif act == "postpone":
        head += f" на {_h(decision['postpone_hours'])}"
        if local_time_str:
            head += f" (≈ {local_time_str})"
        if am:
            head += f", норма {am['gross_mm']:.0f} мм"
        elif decision.get("waiting_for_rain"):
            head += " — чекаємо дощ, потім перерахунок"
    main = [reason_text(r) for r in reasons[:2]]
    summary = head + ". " + " ".join(main)
    summary += (f" Вологість зараз ≈ {100 * moisture['theta']:.0f}% об. "
                f"(±{100 * moisture['theta_sigma']:.0f}%), впевненість {decision['confidence_label']}.")
    return dict(headline=head + _cycles_suffix(decision), summary=summary, details=[reason_text(r) for r in reasons])
