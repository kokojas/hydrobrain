"""Графіки і підсумкова таблиця Етапу 2 (ds/stage2/results)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .experiment import MODEL_UK, MODELS, RES  # noqa: E402

COL = dict(persistence="#9E9E9E", physics="#C77C48", physics_cal="#E0A878", ml_only="#5B7DB1",
           ml_physfeat="#8FA8D0", hybrid="#3A8F6B")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _bars(ax, sub, title):
    sub = sub.set_index("model").loc[MODELS]
    x = np.arange(len(MODELS))
    ax.bar(x, sub["MAE"], color=[COL[m] for m in MODELS])
    for i, v in enumerate(sub["MAE"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [MODEL_UK[m].split(" ", 1)[0] for m in MODELS])
    ax.set_title(title)
    ax.set_ylabel("MAE, % об.")


def fig_main(res):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, h in zip(axes, (1, 2)):
        _bars(ax, res[(res.split == "A_test_2023-25") & (res.h == h)], f"Тест 2023–2025, горизонт {h} доб.")
    fig.tight_layout()
    fig.savefig(RES / "fig1_mae_test.png", dpi=150)
    plt.close(fig)


def fig_forecast(res):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(MODELS))
    for ax, h in zip(axes, (1, 2)):
        a = res[(res.split == "B_2024-25") & (res.h == h) & (res.weather == "ideal")].set_index("model").loc[MODELS]
        b = res[(res.split == "B_2024-25") & (res.h == h) & (res.weather == "fcst")].set_index("model").loc[MODELS]
        ax.bar(x - 0.2, a["MAE"], 0.4, label="факт погоди (ідеал)", color=[COL[m] for m in MODELS], alpha=0.45)
        ax.bar(x + 0.2, b["MAE"], 0.4, label="справжній прогноз", color=[COL[m] for m in MODELS])
        ax.set_xticks(x, [MODEL_UK[m].split(" ", 1)[0] for m in MODELS])
        ax.set_title(f"2024–2025, горизонт {h} доб.")
        ax.set_ylabel("MAE, % об.")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig2_ideal_vs_forecast.png", dpi=150)
    plt.close(fig)


def fig_lolo():
    t = pd.read_csv(RES / "lolo_by_site_h1.csv")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    sites = list(dict.fromkeys(t["site"]))
    x = np.arange(len(sites))
    for k, m in enumerate(["persistence", "physics", "ml_only", "hybrid"]):
        v = t[t.model == m].set_index("site").loc[sites, "MAE"]
        ax.plot(x, v, "o-", color=COL[m], label=MODEL_UK[m], ms=4)
    ax.set_xticks(x, sites, rotation=45, ha="right")
    ax.set_ylabel("MAE, % об.")
    ax.set_title("Нова точка (leave-location-out), тест 2023–2025, горизонт 1 доба")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(RES / "fig3_lolo_sites.png", dpi=150)
    plt.close(fig)


def fig_example(site="poltava", year=2024):
    p = pd.read_csv(RES / "predictions_test.csv.gz", parse_dates=["time"])
    s = p[(p.site == site) & (p.h == 1) & (p.weather == "fcst") & (p.time.dt.year == year)].sort_values("time")
    s = s[(s.time.dt.month >= 5) & (s.time.dt.month <= 8)]
    if s.empty:
        return
    k = s["theta_range"].iloc[0] * 100
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(s.time, s.y * k, color="black", lw=1.2, label="ERA5-Land (правда)")
    for m in ("physics", "ml_only", "hybrid"):
        ax.plot(s.time, s[m] * k, color=COL[m], lw=1, label=MODEL_UK[m])
    ax.axhline(0, color="#BBBBBB", lw=0.5)
    ax.set_ylabel("зміна вологості за добу, % об.")
    ax.set_title(f"{site}, травень–серпень {year}: прогноз на 1 добу за справжнім прогнозом погоди")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(RES / "fig4_example_season.png", dpi=150)
    plt.close(fig)


def _md(df, cols, fmt="{:.3f}"):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def make(res: pd.DataFrame, extras: dict):
    fig_main(res)
    fig_forecast(res)
    fig_lolo()
    fig_example()
    lines = ["# Етап 2 — результати\n",
             "MAE і RMSE — у % об. вологості кореневої зони 0–100 см (ERA5-Land). "
             "MAE_rain — доби з фактичними опадами ≥ 5 мм на горизонті, MAE_dry — < 0.5 мм.\n"]
    for split, title in [("A_test_2023-25", "A. Тест 2023–2025 (ідеальна погода)"),
                         ("B_2024-25", "B. 2024–2025: факт погоди vs справжній прогноз"),
                         ("C_LOLO", "C. Нова точка (leave-location-out), тест 2023–2025"),
                         ("D_France", "D. Інший клімат: Франція (ERA5-Land), модель навчена на Україні")]:
        lines.append(f"\n## {title}\n")
        for h in (1, 2):
            sub = res[(res.split == split) & (res.h == h)].copy()
            sub["model"] = sub["model"].map(MODEL_UK)
            cols = ["model", "weather", "MAE", "RMSE", "R2", "MAE_rain", "MAE_dry", "bias"]
            lines.append(f"\n**Горизонт {h} доб.**\n\n" + _md(sub, cols))
    lines.append("\n## Додатково\n\n```json\n" + json.dumps(extras, indent=1, ensure_ascii=False) + "\n```\n")
    (RES / "summary.md").write_text("".join(lines), encoding="utf-8")
