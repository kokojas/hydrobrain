"""
Довідкові бази даних «мозку» поливу.

Джерела значень:
  * FAO-56 (Allen et al., 1998): Таблиці 11 (тривалість фаз), 12 (Kc), 17 (Kcb),
    19 (REW/TEW), 20 (fw), 22 (Zr, p), 23 (ECe поріг, b), 24 (Ky).
  * Yin et al., 1995 (Agric. For. Meteorol. 77: 1-16) — кардинальні температури
    для бета-функції розвитку (значення для конкретних культур — літературні
    наближення, їх варто калібрувати під сорт).
  * NRCS NEH Part 630, Ch. 9-10 — числа кривої стоку (CN).
  * Типові властивості ґрунтів України — узагальнені середні значення орного шару
    (наближення; для точної роботи краще ввести дані агрохімічного паспорта поля).
"""

# ---------------------------------------------------------------------------
# 1. КУЛЬТУРИ
# ---------------------------------------------------------------------------
# kc_*      — одиночний коефіцієнт культури (FAO-56, Табл. 12)
# kcb_*     — базальний коефіцієнт (FAO-56, Табл. 17)
# h         — максимальна висота, м
# zr        — (мін, макс) глибина коренів, м (Табл. 22)
# p         — частка легкодоступної води (Табл. 22, для ETc ≈ 5 мм/добу)
# stages    — (L_ini, L_dev, L_mid, L_late), діб (Табл. 11, континентальні/європейські рядки)
# ece_thr,b — поріг засолення ECe (дСм/м) та втрата врожаю %/(дСм/м) (Табл. 23)
# ky        — сезонний коефіцієнт реакції врожаю (Табл. 24)
# t_card    — (Tb, To, Tc) кардинальні температури для бета-моделі Yin et al.
# t_typ     — типова середня температура сезону в Україні, за якої фаза триває stages днів
# ph_opt    — оптимальний діапазон pH ґрунту
# sensitive — діапазон стадії розвитку DVS (0..1), коли культура найчутливіша до нестачі води
# cover     — клас покриву для CN (row / small_grain / legume / vegetable / orchard / turf)
# season    — warm / cool (для реакції на холодний ґрунт і холодну воду)
# winter    — озима культура (розвиток рахується за календарем, бо є період спокою)
CROPS = {
    "maize": dict(name_uk="Кукурудза на зерно", kc_ini=0.30, kc_mid=1.20, kc_end=0.60,
                  kcb_ini=0.15, kcb_mid=1.15, kcb_end=0.50, h=2.0, zr=(0.30, 1.40), p=0.55,
                  stages=(30, 40, 50, 30), ece_thr=1.7, b=12.0, ky=1.25,
                  t_card=(8.0, 30.0, 40.0), t_typ=19.0, ph_opt=(6.0, 7.5),
                  sensitive=(0.40, 0.65), cover="row", season="warm", winter=False),
    "sunflower": dict(name_uk="Соняшник", kc_ini=0.35, kc_mid=1.05, kc_end=0.35,
                      kcb_ini=0.15, kcb_mid=1.00, kcb_end=0.25, h=2.0, zr=(0.30, 1.50), p=0.45,
                      stages=(25, 35, 45, 25), ece_thr=4.8, b=5.0, ky=0.95,
                      t_card=(6.0, 28.0, 40.0), t_typ=19.0, ph_opt=(6.0, 7.8),
                      sensitive=(0.40, 0.70), cover="row", season="warm", winter=False),
    "winter_wheat": dict(name_uk="Пшениця озима", kc_ini=0.70, kc_mid=1.15, kc_end=0.30,
                         kcb_ini=0.30, kcb_mid=1.10, kcb_end=0.20, h=1.0, zr=(0.30, 1.50), p=0.55,
                         stages=(160, 75, 75, 25), ece_thr=6.0, b=7.1, ky=1.05,
                         t_card=(0.0, 25.0, 35.0), t_typ=10.0, ph_opt=(6.0, 7.5),
                         sensitive=(0.70, 0.85), cover="small_grain", season="cool", winter=True),
    "spring_barley": dict(name_uk="Ячмінь ярий", kc_ini=0.30, kc_mid=1.15, kc_end=0.25,
                          kcb_ini=0.15, kcb_mid=1.10, kcb_end=0.15, h=1.0, zr=(0.30, 1.20), p=0.55,
                          stages=(15, 25, 50, 30), ece_thr=8.0, b=5.0, ky=1.0,
                          t_card=(0.0, 24.0, 34.0), t_typ=15.0, ph_opt=(6.0, 7.8),
                          sensitive=(0.35, 0.60), cover="small_grain", season="cool", winter=False),
    "soybean": dict(name_uk="Соя", kc_ini=0.40, kc_mid=1.15, kc_end=0.50,
                    kcb_ini=0.15, kcb_mid=1.10, kcb_end=0.30, h=0.8, zr=(0.30, 1.10), p=0.50,
                    stages=(20, 30, 60, 25), ece_thr=5.0, b=20.0, ky=0.85,
                    t_card=(7.0, 30.0, 40.0), t_typ=19.0, ph_opt=(6.0, 7.0),
                    sensitive=(0.45, 0.75), cover="row", season="warm", winter=False),
    "winter_rapeseed": dict(name_uk="Ріпак озимий", kc_ini=0.35, kc_mid=1.07, kc_end=0.35,
                            kcb_ini=0.15, kcb_mid=1.00, kcb_end=0.25, h=1.5, zr=(0.30, 1.30), p=0.60,
                            stages=(150, 60, 55, 30), ece_thr=9.7, b=14.0, ky=1.0,
                            t_card=(0.0, 22.0, 32.0), t_typ=10.0, ph_opt=(6.0, 7.5),
                            sensitive=(0.65, 0.85), cover="small_grain", season="cool", winter=True),
    "sugar_beet": dict(name_uk="Буряк цукровий", kc_ini=0.35, kc_mid=1.20, kc_end=0.70,
                       kcb_ini=0.15, kcb_mid=1.15, kcb_end=0.50, h=0.5, zr=(0.30, 1.20), p=0.55,
                       stages=(45, 40, 55, 40), ece_thr=7.0, b=5.9, ky=1.0,
                       t_card=(3.0, 25.0, 35.0), t_typ=17.0, ph_opt=(6.5, 7.8),
                       sensitive=(0.30, 0.60), cover="row", season="cool", winter=False),
    "potato": dict(name_uk="Картопля", kc_ini=0.50, kc_mid=1.15, kc_end=0.75,
                   kcb_ini=0.15, kcb_mid=1.10, kcb_end=0.65, h=0.6, zr=(0.25, 0.60), p=0.35,
                   stages=(25, 30, 45, 30), ece_thr=1.7, b=12.0, ky=1.10,
                   t_card=(4.0, 22.0, 32.0), t_typ=17.0, ph_opt=(5.0, 6.5),
                   sensitive=(0.35, 0.70), cover="row", season="cool", winter=False),
    "tomato": dict(name_uk="Томат", kc_ini=0.60, kc_mid=1.15, kc_end=0.80,
                   kcb_ini=0.15, kcb_mid=1.10, kcb_end=0.70, h=0.6, zr=(0.30, 1.20), p=0.40,
                   stages=(30, 40, 40, 25), ece_thr=2.5, b=9.0, ky=1.05,
                   t_card=(10.0, 26.0, 35.0), t_typ=20.0, ph_opt=(6.0, 7.0),
                   sensitive=(0.35, 0.65), cover="vegetable", season="warm", winter=False),
    "cucumber": dict(name_uk="Огірок", kc_ini=0.60, kc_mid=1.00, kc_end=0.75,
                     kcb_ini=0.15, kcb_mid=0.95, kcb_end=0.70, h=0.3, zr=(0.25, 1.00), p=0.50,
                     stages=(20, 30, 40, 15), ece_thr=2.5, b=13.0, ky=1.0,
                     t_card=(12.0, 28.0, 38.0), t_typ=21.0, ph_opt=(6.0, 7.0),
                     sensitive=(0.30, 0.70), cover="vegetable", season="warm", winter=False),
    "pepper": dict(name_uk="Перець солодкий", kc_ini=0.60, kc_mid=1.05, kc_end=0.90,
                   kcb_ini=0.15, kcb_mid=1.00, kcb_end=0.80, h=0.7, zr=(0.25, 0.80), p=0.30,
                   stages=(25, 35, 40, 20), ece_thr=1.5, b=14.0, ky=1.1,
                   t_card=(10.0, 27.0, 35.0), t_typ=20.0, ph_opt=(6.0, 7.0),
                   sensitive=(0.35, 0.65), cover="vegetable", season="warm", winter=False),
    "onion": dict(name_uk="Цибуля ріпчаста", kc_ini=0.70, kc_mid=1.05, kc_end=0.75,
                  kcb_ini=0.15, kcb_mid=0.95, kcb_end=0.65, h=0.4, zr=(0.20, 0.50), p=0.30,
                  stages=(15, 25, 70, 40), ece_thr=1.2, b=16.0, ky=1.1,
                  t_card=(4.0, 22.0, 32.0), t_typ=17.0, ph_opt=(6.0, 7.5),
                  sensitive=(0.40, 0.75), cover="vegetable", season="cool", winter=False),
    "cabbage": dict(name_uk="Капуста", kc_ini=0.70, kc_mid=1.05, kc_end=0.95,
                    kcb_ini=0.15, kcb_mid=0.95, kcb_end=0.85, h=0.4, zr=(0.25, 0.70), p=0.45,
                    stages=(40, 60, 50, 15), ece_thr=1.8, b=9.7, ky=0.95,
                    t_card=(4.0, 20.0, 30.0), t_typ=16.0, ph_opt=(6.5, 7.5),
                    sensitive=(0.45, 0.85), cover="vegetable", season="cool", winter=False),
    "carrot": dict(name_uk="Морква", kc_ini=0.70, kc_mid=1.05, kc_end=0.95,
                   kcb_ini=0.15, kcb_mid=0.95, kcb_end=0.85, h=0.3, zr=(0.25, 0.80), p=0.35,
                   stages=(20, 30, 50, 20), ece_thr=1.0, b=14.0, ky=1.0,
                   t_card=(4.0, 22.0, 32.0), t_typ=16.0, ph_opt=(6.0, 7.0),
                   sensitive=(0.40, 0.80), cover="vegetable", season="cool", winter=False),
    "strawberry": dict(name_uk="Полуниця", kc_ini=0.40, kc_mid=0.85, kc_end=0.75,
                       kcb_ini=0.30, kcb_mid=0.80, kcb_end=0.70, h=0.2, zr=(0.20, 0.30), p=0.20,
                       stages=(30, 40, 60, 30), ece_thr=1.0, b=33.0, ky=1.0,
                       t_card=(5.0, 22.0, 32.0), t_typ=17.0, ph_opt=(5.5, 6.5),
                       sensitive=(0.30, 0.60), cover="vegetable", season="cool", winter=False),
    "peas": dict(name_uk="Горох", kc_ini=0.50, kc_mid=1.15, kc_end=1.10,
                 kcb_ini=0.15, kcb_mid=1.10, kcb_end=1.05, h=0.5, zr=(0.25, 0.80), p=0.35,
                 stages=(15, 25, 35, 15), ece_thr=1.5, b=14.0, ky=1.15,
                 t_card=(4.0, 20.0, 30.0), t_typ=15.0, ph_opt=(6.0, 7.5),
                 sensitive=(0.40, 0.70), cover="legume", season="cool", winter=False),
    "green_beans": dict(name_uk="Квасоля овочева", kc_ini=0.50, kc_mid=1.05, kc_end=0.90,
                        kcb_ini=0.15, kcb_mid=1.00, kcb_end=0.80, h=0.4, zr=(0.25, 0.70), p=0.45,
                        stages=(20, 30, 30, 10), ece_thr=1.0, b=19.0, ky=1.15,
                        t_card=(8.0, 27.0, 37.0), t_typ=20.0, ph_opt=(6.0, 7.0),
                        sensitive=(0.40, 0.75), cover="vegetable", season="warm", winter=False),
    "watermelon": dict(name_uk="Кавун", kc_ini=0.40, kc_mid=1.00, kc_end=0.75,
                       kcb_ini=0.15, kcb_mid=0.95, kcb_end=0.70, h=0.4, zr=(0.30, 1.50), p=0.40,
                       stages=(20, 30, 30, 30), ece_thr=2.0, b=10.0, ky=1.1,
                       t_card=(12.0, 30.0, 40.0), t_typ=21.0, ph_opt=(6.0, 7.0),
                       sensitive=(0.35, 0.65), cover="vegetable", season="warm", winter=False),
    "grape": dict(name_uk="Виноград", kc_ini=0.30, kc_mid=0.80, kc_end=0.45,
                  kcb_ini=0.15, kcb_mid=0.75, kcb_end=0.40, h=1.8, zr=(1.00, 1.80), p=0.40,
                  stages=(20, 40, 120, 60), ece_thr=1.5, b=9.6, ky=0.85,
                  t_card=(10.0, 28.0, 38.0), t_typ=18.0, ph_opt=(6.0, 8.0),
                  sensitive=(0.20, 0.45), cover="orchard", season="warm", winter=False),
    "apple": dict(name_uk="Яблуня (сад)", kc_ini=0.45, kc_mid=0.95, kc_end=0.70,
                  kcb_ini=0.35, kcb_mid=0.90, kcb_end=0.65, h=4.0, zr=(1.00, 1.80), p=0.50,
                  stages=(20, 70, 90, 30), ece_thr=1.0, b=18.0, ky=1.0,
                  t_card=(5.0, 23.0, 33.0), t_typ=16.0, ph_opt=(6.0, 7.0),
                  sensitive=(0.30, 0.60), cover="orchard", season="cool", winter=False),
    "lawn": dict(name_uk="Газон (холодостійкі трави)", kc_ini=0.90, kc_mid=0.95, kc_end=0.95,
                 kcb_ini=0.85, kcb_mid=0.90, kcb_end=0.90, h=0.10, zr=(0.30, 0.60), p=0.40,
                 stages=(10, 10, 300, 10), ece_thr=4.0, b=5.0, ky=1.0,
                 t_card=(4.0, 20.0, 32.0), t_typ=15.0, ph_opt=(6.0, 7.5),
                 sensitive=(0.0, 0.0), cover="turf", season="cool", winter=False),
}

# ---------------------------------------------------------------------------
# 2. ҐРУНТИ УКРАЇНИ (орний шар 0-30 см, середні значення)
# ---------------------------------------------------------------------------
# sand/clay — % маси; om — органічна речовина, %; bd — щільність, г/см³;
# ph — pH(H2O); ece — базова ECe, дСм/м; depth — ефективна глибина профілю, м;
# gw — типова глибина ґрунтових вод, м (None — глибоко, капілярне підживлення = 0);
# zone — де найпоширеніші
# fc/wp — ВИМІРЯНІ агрогідрологічні константи (найменша вологоємність і вологість в'янення), перераховані
# з % маси в м³/м³ через щільність кожного шару і усереднені для шару 0–100 см (ваги шарів 0.2/0.3/0.25/0.25
# або 0.2/0.3/0.2/0.3 — за глибинами в таблицях). Мають пріоритет над педотрансфером, коли тип ґрунту
# вибрано з довідника. Джерела:
#   [Д65]  Довідник агрогідрологічних властивостей ґрунтів Української РСР. К., 1965 (табл. 3.44 у підручнику
#          з ґрунтознавства) — чорноземи типові;
#   [П79]  Полупан М.І., Кисіль В.Д., Гаврик П.О., 1979 (табл. 3.52 «Водно-фізичні показники чорноземів Степу»).
# Для решти ґрунтів вимірювань у доступних джерелах не знайдено — для них fc/wp = None, і модель
# рахує константи педотрансфером Saxton & Rawls (2006).
SOILS_UA = {
    "chornozem_typical": dict(name_uk="Чорнозем типовий (важкосуглинковий)", sand=15, clay=30, om=4.5, bd=1.20,
                              ph=7.0, ece=0.4, depth=1.5, gw=None, structure="good",
                              fc=0.408, wp=0.235, src="[Д65]: НВ 39.0→27.2 %, ВВ 16.9→17.9 % маси, щільність 1.13→1.46",
                              zone="Лісостеп (Полтавщина, Черкащина, Вінниччина)"),
    "chornozem_typical_medium": dict(name_uk="Чорнозем типовий (середньосуглинковий)", sand=35, clay=18, om=3.5,
                                     bd=1.17, ph=6.9, ece=0.3, depth=1.4, gw=None, structure="good",
                                     fc=0.270, wp=0.120,
                                     src="[Д65]: НВ 26.7→20.0 %, ВВ 9.9→8.9 % маси, щільність 1.13→1.23",
                                     zone="Лісостеп, легші за гранулометричним складом відміни"),
    "chornozem_podzolized": dict(name_uk="Чорнозем опідзолений", sand=20, clay=25, om=3.2, bd=1.28,
                                 ph=6.2, ece=0.3, depth=1.3, gw=None, structure="good",
                                 zone="Північний Лісостеп"),
    "chornozem_ordinary": dict(name_uk="Чорнозем звичайний", sand=18, clay=33, om=4.0, bd=1.25,
                               ph=7.2, ece=0.5, depth=1.4, gw=None, structure="good",
                               fc=0.306, wp=0.150, src="[П79]: малогумусний на лесі, НВ 29.8→22.6 %, ВВ 12.1→11.7 % маси, щільність 1.17→1.36",
                               zone="Північний Степ (Дніпропетровщина, Кіровоградщина)"),
    "chornozem_southern": dict(name_uk="Чорнозем південний", sand=22, clay=32, om=3.0, bd=1.30,
                               ph=7.5, ece=0.8, depth=1.2, gw=None, structure="medium",
                               fc=0.332, wp=0.150, src="[П79]: на лесі, НВ 31.9→23.3 %, ВВ 11.0→12.3 % маси, щільність 1.00→1.42",
                               zone="Південний Степ (Миколаївщина, Одещина, Запоріжжя)"),
    "meadow_chornozem": dict(name_uk="Лучно-чорноземний", sand=15, clay=35, om=4.0, bd=1.25,
                             ph=7.4, ece=0.8, depth=1.5, gw=2.0, structure="good",
                             zone="Тераси річок, знижені ділянки"),
    "grey_forest": dict(name_uk="Сірий лісовий", sand=28, clay=20, om=2.0, bd=1.35,
                        ph=5.8, ece=0.2, depth=1.2, gw=None, structure="medium",
                        zone="Правобережний Лісостеп, Житомирщина"),
    "sod_podzolic": dict(name_uk="Дерново-підзолистий (супіщаний)", sand=70, clay=8, om=1.2, bd=1.45,
                         ph=5.2, ece=0.15, depth=0.9, gw=2.5, structure="poor",
                         zone="Полісся (Волинь, Рівненщина, Чернігівщина)"),
    "dark_chestnut": dict(name_uk="Темно-каштановий", sand=25, clay=32, om=2.5, bd=1.35,
                          ph=7.8, ece=1.2, depth=1.0, gw=None, structure="medium",
                          zone="Сухий Степ (Херсонщина, Запоріжжя)"),
    "chestnut_solonetzic": dict(name_uk="Каштановий солонцюватий", sand=25, clay=36, om=1.8, bd=1.45,
                                ph=8.2, ece=2.5, depth=0.9, gw=None, structure="poor",
                                zone="Південь Херсонщини, Присивашшя"),
    "sandy_soddy": dict(name_uk="Дерновий піщаний / піски", sand=88, clay=4, om=0.7, bd=1.55,
                        ph=6.0, ece=0.1, depth=1.0, gw=3.0, structure="poor",
                        zone="Нижньодніпровські піски, Полісся"),
    "alluvial_meadow": dict(name_uk="Лучний алювіальний", sand=30, clay=22, om=2.8, bd=1.30,
                            ph=7.0, ece=0.6, depth=1.2, gw=1.5, structure="medium",
                            zone="Заплави річок"),
}

# ---------------------------------------------------------------------------
# 3. ГЕОЛОГІЯ (класи материнської/корінної породи)
# ---------------------------------------------------------------------------
# ds_max — макс. глибина ґрунту над породою, м; drainage — клас дренажу;
# ece_add — додаток до базової ECe, дСм/м; ph_shift — зсув pH;
# taw_sigma — невизначеність запасу вологи для цього класу (σ_geo)
GEOLOGY = {
    "loess": dict(name_uk="Лес / лесоподібні суглинки", ds_max=3.0, drainage="good",
                  ece_add=0.1, ph_shift=0.2, taw_sigma=0.12,
                  keywords=["loess", "лес", "silt", "aeolian"]),
    "sand": dict(name_uk="Піски (алювіальні, флювіогляціальні)", ds_max=3.0, drainage="excessive",
                 ece_add=0.0, ph_shift=-0.3, taw_sigma=0.20,
                 keywords=["sand", "пісок", "піски", "fluvioglacial", "glaciofluvial"]),
    "clay": dict(name_uk="Глини, моренні суглинки", ds_max=2.0, drainage="poor",
                 ece_add=0.3, ph_shift=0.1, taw_sigma=0.18,
                 keywords=["clay", "глина", "till", "moraine", "mudstone", "shale", "argill"]),
    "carbonate": dict(name_uk="Вапняки, мергелі, крейда", ds_max=1.2, drainage="good",
                      ece_add=0.4, ph_shift=0.6, taw_sigma=0.20,
                      keywords=["limestone", "marl", "chalk", "carbonate", "dolomit", "вапняк", "мергель", "крейда"]),
    "crystalline": dict(name_uk="Кристалічні породи (Український щит)", ds_max=0.8, drainage="moderate",
                        ece_add=0.0, ph_shift=-0.2, taw_sigma=0.25,
                        keywords=["granite", "gneiss", "migmatite", "crystalline", "metamorphic",
                                  "граніт", "гнейс", "plutonic", "precambrian", "archean", "proterozoic"]),
    "alluvium": dict(name_uk="Алювій (заплавні відклади)", ds_max=2.5, drainage="moderate",
                     ece_add=0.3, ph_shift=0.0, taw_sigma=0.20,
                     keywords=["alluvi", "fluvial", "floodplain", "алювій", "holocene"]),
    "saline": dict(name_uk="Засолені відклади (лиманні, морські)", ds_max=2.0, drainage="poor",
                   ece_add=2.0, ph_shift=0.8, taw_sigma=0.25,
                   keywords=["saline", "evaporite", "gypsum", "salt", "lagoon", "marine", "лиман"]),
    "unknown": dict(name_uk="Невідомо", ds_max=2.0, drainage="moderate", ece_add=0.2,
                    ph_shift=0.0, taw_sigma=0.25, keywords=[]),
}

DRAINAGE_CLASSES = {  # множник швидкості дренажу τ і зсув CN
    "excessive": dict(name_uk="Надмірний (піски)", tau_mult=1.4, cn_shift=-4),
    "good": dict(name_uk="Добрий", tau_mult=1.0, cn_shift=0),
    "moderate": dict(name_uk="Помірний", tau_mult=0.7, cn_shift=+2),
    "poor": dict(name_uk="Поганий", tau_mult=0.35, cn_shift=+5),
    "very_poor": dict(name_uk="Дуже поганий (перезволоження)", tau_mult=0.15, cn_shift=+8),
}

# ---------------------------------------------------------------------------
# 4. СПОСОБИ ПОЛИВУ
# ---------------------------------------------------------------------------
# ea — ефективність застосування (частка води, що потрапляє в кореневу зону);
# fw — частка змоченої поверхні (FAO-56, Табл. 20); sigma — невизначеність норми;
# wind_sensitive — чи страждає від вітру (знос крапель); min_mm — мінімальна разова норма
IRRIGATION_METHODS = {
    "drip": dict(name_uk="Крапельний", ea=0.90, fw=0.35, sigma=0.05, wind_sensitive=False, min_mm=2.0, max_mm=35),
    "subsurface_drip": dict(name_uk="Підґрунтовий крапельний", ea=0.95, fw=0.05, sigma=0.05,
                            wind_sensitive=False, min_mm=2.0, max_mm=35),
    "micro_sprinkler": dict(name_uk="Мікродощування", ea=0.85, fw=0.55, sigma=0.08,
                            wind_sensitive=True, min_mm=3.0, max_mm=30),
    "sprinkler": dict(name_uk="Дощування (стаціонарне)", ea=0.75, fw=1.0, sigma=0.12,
                      wind_sensitive=True, min_mm=5.0, max_mm=40),
    "center_pivot": dict(name_uk="Кругова дощувальна машина", ea=0.85, fw=1.0, sigma=0.10,
                         wind_sensitive=True, min_mm=8.0, max_mm=30),
    "furrow": dict(name_uk="По борознах", ea=0.60, fw=0.70, sigma=0.20, wind_sensitive=False, min_mm=30.0, max_mm=100),
    "basin": dict(name_uk="Затоплення чеків", ea=0.55, fw=1.0, sigma=0.20, wind_sensitive=False, min_mm=40.0, max_mm=120),
    "hose": dict(name_uk="Шланг / вручну", ea=0.65, fw=1.0, sigma=0.25, wind_sensitive=False, min_mm=3.0, max_mm=25),
}

# ---------------------------------------------------------------------------
# 5. ДОБРИВА — сольовий індекс (NaNO3 = 100) → kFt = SI/100
# ---------------------------------------------------------------------------
FERTILIZERS = {
    "none": dict(name_uk="Без добрив", si=0.0),
    "ammonium_nitrate": dict(name_uk="Аміачна селітра", si=105.0),
    "urea": dict(name_uk="Карбамід (сечовина)", si=75.0),
    "uan": dict(name_uk="КАС (карбамідно-аміачна суміш)", si=75.0),
    "calcium_nitrate": dict(name_uk="Кальцієва селітра", si=53.0),
    "potassium_chloride": dict(name_uk="Хлористий калій", si=116.0),
    "potassium_sulfate": dict(name_uk="Сульфат калію", si=46.0),
    "map": dict(name_uk="Моноамонійфосфат (MAP)", si=30.0),
    "dap": dict(name_uk="Діамонійфосфат (DAP)", si=29.0),
    "npk_complex": dict(name_uk="Комплексне NPK", si=50.0),
    "organic": dict(name_uk="Органічні (гній, компост)", si=10.0),
}

# ---------------------------------------------------------------------------
# 6. ЧИСЛА КРИВОЇ СТОКУ CN для ARC II, гідрологічні групи A/B/C/D
# (NEH-630 Ch.9, Табл. 9-1, «добрий» гідрологічний стан)
# ---------------------------------------------------------------------------
CN_TABLE = {
    "fallow": (77, 86, 91, 94),        # чорний пар
    "row": (67, 78, 85, 89),           # просапні, прямі рядки
    "row_residue": (64, 75, 82, 85),   # просапні з рослинними рештками
    "small_grain": (63, 75, 83, 87),   # зернові колосові
    "legume": (58, 72, 81, 85),        # бобові суцільної сівби
    "vegetable": (67, 78, 85, 89),
    "orchard": (43, 65, 76, 82),       # сад/виноградник з міжряддями під травою
    "turf": (39, 61, 74, 80),          # газон/пасовище, добрий стан
}

# Ґрунтова REW/TEW-орієнтири за текстурою (FAO-56 Табл. 19), REW у мм
REW_BY_TEXTURE = {
    "sand": 4.5, "loamy sand": 6.0, "sandy loam": 8.0, "loam": 9.0, "silt loam": 9.5,
    "silt": 9.5, "silty clay loam": 9.5, "silty clay": 10.0, "clay": 10.0,
    "sandy clay loam": 9.0, "clay loam": 9.5, "sandy clay": 10.0,
}

# Максимальне капілярне підживлення (мм/добу) з неглибоких ґрунтових вод за текстурою
CR_MAX_BY_TEXTURE = {
    "sand": 1.5, "loamy sand": 2.5, "sandy loam": 4.0, "loam": 5.0, "silt loam": 6.0,
    "silt": 6.5, "silty clay loam": 4.5, "silty clay": 3.0, "clay": 2.5,
    "sandy clay loam": 4.0, "clay loam": 4.0, "sandy clay": 2.5,
}
