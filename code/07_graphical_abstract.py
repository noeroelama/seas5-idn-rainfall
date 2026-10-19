"""
Graphical abstract (submitted as a separate file), 13.5 cm wide:
  (1) MAE skill of quantile-mapped SEAS5 (B4) relative to climatology per province, lead 1, all months
  (2) MAE skill of gradient boosting M2 and of the average of B4 and M2 relative to B4, by lead, with 90% intervals
  (3) MAE skill added by each group of predictors at lead 1 (M2 vs M1, M3 vs M2, M4 vs M3), with 90% intervals
Output: figures/graphical_abstract.pdf and .png (300 dpi).
Usage: python code/07_graphical_abstract.py
"""
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.colors import LinearSegmentedColormap     # noqa: E402

from config import FIGURES_DIR, LEADS, OBS_PANEL, PROVINCE_NAME_EN, PROVINCE_POLYGONS, RESULTS_DIR  # noqa: E402

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EQM, GB, AVERAGE, NEUTRAL = "#1baf7a", "#e87ba4", "#2a78d6", "#a8a7a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
                     "pdf.fonttype": 42})
GREENS = LinearSegmentedColormap.from_list("skill", ["#f3f7f4", "#9fd8bf", EQM, "#0b6b48"])

# ------------------------------------------------------------------ data
forecasts = pd.read_parquet(RESULTS_DIR / "forecasts_amount.parquet", columns=["province_idx", "lead", "y", "B0", "B4"])
lead1 = forecasts[forecasts["lead"] == 1].assign(e0=lambda d: (d.y - d.B0).abs(), e4=lambda d: (d.y - d.B4).abs())
mae = lead1.groupby("province_idx")[["e0", "e4"]].mean()
provinces = sorted(pd.read_csv(OBS_PANEL).province.unique())
province_skill = pd.DataFrame({"province": provinces, "skill": 100 * (1 - mae.e4 / mae.e0).values})
overall_skill = 100 * (1 - lead1.e4.mean() / lead1.e0.mean())
polygons = gpd.read_file(PROVINCE_POLYGONS)
polygons["province"] = polygons.PROVINSI.map(PROVINCE_NAME_EN)
polygons = polygons.merge(province_skill, on="province")

gain = pd.read_csv(RESULTS_DIR / "gain_vs_simple.csv")
gain = gain[(gain.season == "all") & (gain.reference == "B4")]
ladder = pd.read_csv(RESULTS_DIR / "ladder.csv")
ladder = ladder[(ladder.season == "all") & (ladder["lead"] == 1)].set_index("forecast")

# ------------------------------------------------------------------ layout
fig = plt.figure(figsize=(13.5 / 2.54, 7.4 / 2.54))
fig.text(0.012, 0.97, "Does gradient boosting improve quantile-mapped SEAS5 rainfall forecasts?",
         fontsize=8.5, fontweight="bold", color=INK, va="top")
fig.text(0.012, 0.895, "38 provinces of Indonesia  |  test years 2005–2024  |  leads 1–6 months  |  "
         "models trained only on earlier data", fontsize=6.3, color=INK_2, va="top")
TITLE_Y = 0.80
ax_map = fig.add_axes([0.01, 0.29, 0.36, 0.44])
ax_bar = fig.add_axes([0.06, 0.20, 0.26, 0.032])
ax_gain = fig.add_axes([0.455, 0.17, 0.255, 0.56])
ax_info = fig.add_axes([0.81, 0.17, 0.185, 0.56])


def panel_title(x, number, text):
    fig.text(x, TITLE_Y, number, fontsize=8, fontweight="bold", color="white", va="center", ha="center",
             bbox=dict(boxstyle="circle,pad=0.18", fc=INK, ec="none"))
    fig.text(x + 0.022, TITLE_Y, text, fontsize=7, fontweight="bold", color=INK, va="center")


# (1) map of B4 skill
norm = plt.Normalize(0, 40)
polygons.plot(ax=ax_map, column="skill", cmap=GREENS, norm=norm, edgecolor="white", linewidth=0.25)
ax_map.set_xlim(94.5, 141.5)
ax_map.set_ylim(-11.5, 6.5)
ax_map.set_aspect("equal")
ax_map.axis("off")
panel_title(0.025, "1", "Quantile-mapped SEAS5 (EQM)\nvs climatology, lead 1")
colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=GREENS, norm=norm), cax=ax_bar, orientation="horizontal",
                        extend="max")
colorbar.set_ticks([0, 10, 20, 30, 40])
colorbar.ax.tick_params(labelsize=6, length=2)
colorbar.outline.set_linewidth(0.4)
colorbar.set_label(f"MAE skill (%); all provinces {overall_skill:.0f}%", fontsize=6.3, labelpad=1)
fig.text(0.012, 0.025, "Map lines delineate study areas and do not necessarily depict accepted national boundaries.",
         fontsize=4.8, color=INK_2)

# (2) gradient boosting and the average, relative to EQM
x = np.array(LEADS)
for offset, key, label, color in ((-0.17, "M2", "Gradient boosting (M2)", GB),
                                  (0.17, "C_B4_M2", "Average of EQM and M2", AVERAGE)):
    d = gain[gain.forecast == key].set_index("lead").loc[list(LEADS)]
    ax_gain.bar(x + offset, d.skill, width=0.32, color=color, label=label, zorder=2)
    ax_gain.errorbar(x + offset, d.skill, yerr=[d.skill - d.lo, d.hi - d.skill], fmt="none", ecolor=INK_2,
                     elinewidth=0.6, capsize=1.2, zorder=3)
ax_gain.axhline(0, color=INK, lw=0.8, zorder=1)
ax_gain.set_xticks(x)
ax_gain.set_xlim(0.5, 6.5)
ax_gain.set_xlabel("Lead (months)", labelpad=1)
ax_gain.set_ylabel("MAE skill vs EQM (%)", labelpad=1)
ax_gain.set_ylim(-3.0, 3.2)
ax_gain.grid(axis="x", visible=False)
ax_gain.text(1.03, 0.64, "better \u2192", transform=ax_gain.transAxes, rotation=90, fontsize=6, color=INK_2,
             ha="left", va="center")
ax_gain.text(1.03, 0.22, "\u2190 worse", transform=ax_gain.transAxes, rotation=90, fontsize=6, color=INK_2,
             ha="left", va="center")
ax_gain.legend(loc="upper right", fontsize=6, handlelength=1.0, handleheight=0.8, borderaxespad=0.1)
panel_title(0.395, "2", "Gradient boosting and the\naverage, relative to EQM")

# (3) what each group of predictors adds at lead 1
steps = [("M2", "Recent\nrain"), ("M3", "Neigh-\nbours"), ("M4", "Niño3.4\nDMI")]
xs = np.arange(len(steps))
values = np.array([ladder.loc[k, "skill"] for k, _ in steps])
lo = np.array([ladder.loc[k, "lo"] for k, _ in steps])
hi = np.array([ladder.loc[k, "hi"] for k, _ in steps])
ax_info.bar(xs, values, width=0.6, color=[GB if low > 0 else NEUTRAL for low in lo], zorder=2)
ax_info.errorbar(xs, values, yerr=[values - lo, hi - values], fmt="none", ecolor=INK_2, elinewidth=0.6, capsize=1.2,
                 zorder=3)
for xi, v, h in zip(xs, values, hi):
    ax_info.text(xi, max(h, 0) + 0.08, f"{v:+.1f}".replace("-", "\u2212"), ha="center", va="bottom", fontsize=6, color=INK)
ax_info.axhline(0, color=INK, lw=0.8, zorder=1)
ax_info.set_xticks(xs)
ax_info.set_xticklabels([label for _, label in steps], fontsize=5.8)
ax_info.set_ylim(-1.0, 1.9)
ax_info.set_yticks([-1, 0, 1])
ax_info.grid(axis="x", visible=False)
ax_info.set_ylabel("MAE skill added (%)", labelpad=1)
panel_title(0.765, "3", "Added predictors,\nlead 1")

for ext in ("pdf", "png"):
    fig.savefig(FIGURES_DIR / f"graphical_abstract.{ext}", dpi=300)
plt.close(fig)
