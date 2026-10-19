"""
Figures of the manuscript, from the outputs of 03_forecasts.py and 04_verify.py (PDF and 300 dpi PNG in figures/).
  Fig1_neighbours     the 38 provinces with links to their four nearest provinces
  Fig2_design         workflow, rolling-origin test and timing of a lead-3 forecast
  Fig3_skill_lead     MAE skill relative to climatology by lead: all months, June-November, December-May
  Fig4_annual         MAE skill relative to climatology in each test year (leads 1 and 3) and the share of dry and wet months
  Fig5_gain           MAE skill of M2 and M4 relative to the best simple correction by lead and season, with 90% intervals
  Fig6_province       MAE skill of M4 relative to EQM (B4) per province, leads 1 and 6
  Fig7_rpss           RPSS relative to climatology, and of M2 and M4 relative to B4
  Fig8_reliability    reliability diagrams for below- and above-normal probabilities, B4 and M4, leads 1 and 3
  Fig9_ladder         MAE skill of each gradient boosting model relative to the model before it (M1: relative to B4)
Usage: python code/05_figures.py
"""
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch                  # noqa: E402

from config import FIGURES_DIR, LEADS, PROVINCE_CENTROIDS, PROVINCE_NAME_EN, PROVINCE_POLYGONS, RESULTS_DIR  # noqa: E402

FIGURES_DIR.mkdir(exist_ok=True)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42})
# one colour and marker per forecast, the same in every figure
COLOR = {"B1": "#2a78d6", "B3": "#eb6834", "B4": "#1baf7a", "B5": "#eda100", "M2": "#e87ba4", "M4": "#008300"}
MARKER = {"B1": "D", "B3": "v", "B4": "o", "B5": "P", "M2": "s", "M4": "^"}
LABEL = {"B1": "Damped anomaly persistence (B1)", "B3": "SEAS5, mean-bias correction (B3)", "B4": "SEAS5, EQM (B4)",
         "B5": "SEAS5, linear MOS (B5)", "M2": "Gradient boosting M2", "M4": "Gradient boosting M4"}
DRY, WET = "#e34948", "#2a78d6"
DIVERGING = LinearSegmentedColormap.from_list("red_grey_blue", [DRY, SURFACE, WET])
SEASONS = [("all", "All months"), ("JunNov", "June–November"), ("DecMay", "December–May")]

polygons = gpd.read_file(PROVINCE_POLYGONS)
polygons["province"] = polygons.PROVINSI.map(PROVINCE_NAME_EN)
centroids = pd.read_csv(PROVINCE_CENTROIDS)
centroids = centroids[centroids.country == "IDN"].set_index("name")


def save(fig, name, **kwargs):
    fig.savefig(FIGURES_DIR / f"{name}.pdf", **kwargs)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, **kwargs)
    plt.close(fig)


def format_map_axes(ax):
    ax.set_xlim(94.5, 141.5)
    ax.set_ylim(-11.5, 6.5)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks(range(95, 145, 10))
    ax.set_xticklabels([f"{x}°E" for x in range(95, 145, 10)])
    ax.set_yticks([-10, -5, 0, 5])
    ax.set_yticklabels(["10°S", "5°S", "0°", "5°N"])
    ax.set_xlabel("")
    ax.set_ylabel("")


def format_lead_axis(ax):
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(list(LEADS))
    ax.set_xlim(0.5, 6.5)
    ax.set_xlabel("Lead time (months)")


def plot_interval(ax, x, estimate, lower, upper, key, label=None):
    """Point estimate with a vertical 90% interval."""
    ax.errorbar(x, estimate, yerr=[estimate - lower, upper - estimate], fmt=MARKER[key], color=COLOR[key], ms=5,
                lw=1.6, capsize=0, label=label)


def legend_below(fig, ax, ncol, y=0.0):
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=ncol, fontsize=8, bbox_to_anchor=(0.5, y))


# ------------------------------------------------------------------ Figure 1: neighbours
neighbours = pd.read_csv(RESULTS_DIR / "neighbours_k4.csv")
fig, ax = plt.subplots(figsize=(7.2, 3.2))
polygons.plot(ax=ax, color=GRID, edgecolor="#ffffff", linewidth=0.4)
for p, q in zip(neighbours.province, neighbours.neighbour):
    ax.plot([centroids.loc[p, "lon"], centroids.loc[q, "lon"]], [centroids.loc[p, "lat"], centroids.loc[q, "lat"]],
            color="#2a78d6", lw=0.7, alpha=0.8, zorder=2)
ax.scatter(centroids.loc[polygons.province, "lon"], centroids.loc[polygons.province, "lat"], s=10, color="#0d366b", zorder=3)
format_map_axes(ax)
fig.tight_layout()
save(fig, "Fig1_neighbours")

# ------------------------------------------------------------------ Figure 2: study design
fig = plt.figure(figsize=(7.2, 6.8))
grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.35, 1.0], hspace=0.22, wspace=0.12)

# (a) workflow
ax = fig.add_subplot(grid[0, :])
ax.set_axis_off()
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)


def box(x, y, w, h, text, fill=SURFACE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=INK_2, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color=INK, linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.9, shrinkA=0, shrinkB=0))


for x, title in ((9, "Data"), (31.5, "Province means"), (58, "Forecasts"), (89, "Verification")):
    ax.text(x, 44.5, title, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
box(0, 30, 18, 11, "SEAS5, 1°\n25 or 51 members\nleads 1–6")
box(0, 16, 18, 11, "CHIRPS v2.0\n0.05°, monthly")
box(0, 2, 18, 11, "Niño3.4 and DMI\nmonthly")
box(23, 16, 17, 25, "Mean for\neach of the\n38 provinces,\n1985–2024")
box(45, 27, 26, 14, "References B0–B5\nclimatology, persistence,\nraw SEAS5, mean bias,\nEQM, linear MOS", fill="#e3f4ec")
box(45, 2, 26, 21, "LightGBM gradient boosting\ncorrection to B3 (M1–M4)\nor to B4 (M1q–M4q)\nM1 SEAS5 predictors\nM2 + recent rainfall\nM3 + neighbours\nM4 + Niño3.4, DMI", fill="#e1efe1")
box(78, 2, 22, 39, "Test years\n2005–2024\n\nAmounts:\nMAE skill,\nDiebold–Mariano,\nyear bootstrap\n\nTerciles:\nRPSS, reliability\n\nChecks: tuning,\nseeds, averages")
arrow(18, 35.5, 23, 35.5)
arrow(18, 21.5, 23, 21.5)
arrow(40, 34, 45, 34)
arrow(40, 19, 45, 15)
arrow(18, 7.5, 45, 7.5)
arrow(58, 27, 58, 23)
ax.text(58.6, 25, "B3, B4", fontsize=6.5, color=INK_2, va="center")
arrow(71, 34, 78, 34)
arrow(71, 12.5, 78, 12.5)
ax.set_title("(a) Workflow", loc="left", fontsize=9)

# (b) rolling-origin test
ax = fig.add_subplot(grid[1, 0])
test_years = list(range(2005, 2025))
for i, T in enumerate(test_years):
    yy = len(test_years) - 1 - i
    ax.barh(yy, T - 3 - 1986, left=1986, height=0.72, color="#c7ddf5", lw=0)
    ax.barh(yy, 3, left=T - 3, height=0.72, color="#6aa3e6", lw=0)
    ax.barh(yy, 1, left=T, height=0.72, color="#eb6834", lw=0)
shown = [2005, 2010, 2015, 2020, 2024]
ax.set_yticks([len(test_years) - 1 - test_years.index(T) for T in shown])
ax.set_yticklabels([str(T) for T in shown])
ax.set_ylabel("Test year T")
ax.set_xlim(1985, 2026)
ax.set_xticks([1986, 2005, 2024])
ax.grid(axis="y", visible=False)
ax.legend(handles=[Patch(color="#c7ddf5", label="Training"), Patch(color="#6aa3e6", label="Training, early stopping"),
                   Patch(color="#eb6834", label="Test")],
          loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, fontsize=7, handlelength=1.2, columnspacing=1.0)
ax.set_title("(b) Rolling-origin test", loc="left", fontsize=9)

# (c) timing of a lead-3 forecast for September
ax = fig.add_subplot(grid[1, 1])
ax.set_axis_off()
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(-3.2, 2.2)
months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
xs = np.linspace(0, 4, len(months))
for x, m in zip(xs, months):
    ax.add_patch(FancyBboxPatch((x - 0.3, -0.3), 0.6, 0.6, boxstyle="round,pad=0,rounding_size=0.08",
                                fc={"Apr": "#c7ddf5", "May": "#c7ddf5", "Jun": "#c7ddf5", "Sep": "#eb6834"}.get(m, SURFACE),
                                ec=INK_2, lw=0.8))
    ax.text(x, 0, m, ha="center", va="center", fontsize=7.5, color=INK)
ax.annotate("", xy=(xs[3] - 0.36, 0.9), xytext=(xs[3] - 0.36, 0.35), arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.9))
ax.text(xs[3] - 0.36, 1.05, "SEAS5 starts 1 July\n(month t − L + 1)", ha="center", va="bottom", fontsize=7, color=INK)
ax.plot([xs[0] - 0.3, xs[2] + 0.3], [-0.55, -0.55], color=INK_2, lw=0.9)
ax.text(xs[1], -0.75, "observed rainfall and indices\nup to end of June (month t − L)", ha="center", va="top", fontsize=7, color=INK)
ax.annotate("", xy=(xs[5], -0.35), xytext=(xs[5], -1.45), arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.9))
ax.text(xs[5], -1.55, "target month t\n(September)", ha="center", va="top", fontsize=7, color=INK)
ax.text(2.0, -2.75, "Lead L = 3: lead 1 is the month in which SEAS5 starts", ha="center", va="center", fontsize=7, color=INK_2)
ax.set_title("(c) Information available for a lead-3 forecast", loc="left", fontsize=9)
save(fig, "Fig2_design", bbox_inches="tight")

# ------------------------------------------------------------------ Figure 3: skill by lead
skill = pd.read_csv(RESULTS_DIR / "skill_amount.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for ax, panel, (season, title) in zip(axs, "abc", SEASONS):
    for key in ("B1", "B3", "B4", "B5", "M2", "M4"):
        d = skill[(skill.season == season) & (skill.method == key)].sort_values("lead")
        ax.plot(d["lead"], d.skill, color=COLOR[key], lw=2, marker=MARKER[key], ms=4, label=LABEL[key])
    format_lead_axis(ax)
    ax.set_title(f"({panel}) {title}", loc="left", fontsize=9)
axs[0].set_ylabel("MAE skill vs climatology (%)")
legend_below(fig, axs[0], ncol=3, y=-0.02)
fig.tight_layout(rect=(0, 0.13, 1, 1))
save(fig, "Fig3_skill_lead")

# ------------------------------------------------------------------ Figure 4: skill in each test year
annual = pd.read_csv(RESULTS_DIR / "annual_skill.csv")
fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
for ax, panel, lead in zip(axs[:2], "ab", (1, 3)):
    d = annual[annual["lead"] == lead]
    for key in ("B4", "M2", "M4"):
        ax.plot(d.year, d[key], color=COLOR[key], lw=1.6, marker=MARKER[key], ms=4, label=LABEL[key])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("MAE skill vs\nclimatology (%)")
    ax.set_title(f"({panel}) Lead {lead}", loc="left", fontsize=9)
d = annual[annual["lead"] == 1]
ax = axs[2]
ax.bar(d.year - 0.2, d.pct_below, width=0.38, color=DRY, label="Below normal (dry)")
ax.bar(d.year + 0.2, d.pct_above, width=0.38, color=WET, label="Above normal (wet)")
ax.axhline(100 / 3, color=INK_2, lw=0.8, ls="--")
ax.set_ylabel("Province-months (%)")
ax.set_ylim(0, 92)
ax.set_yticks([0, 20, 40, 60])
ax.set_title("(c) Share of province-months below and above the training-year terciles", loc="left", fontsize=9)
ax.legend(loc="upper left", ncol=2, fontsize=7.5)
ax.set_xticks(range(2005, 2025, 2))
ax.set_xlim(2004.4, 2024.6)
axs[0].set_ylim(-5, 62)
axs[0].legend(loc="upper left", ncol=3, fontsize=7.5)
fig.tight_layout()
save(fig, "Fig4_annual")

# ------------------------------------------------------------------ Figure 5: gradient boosting relative to B4
gain = pd.read_csv(RESULTS_DIR / "gain_vs_simple.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for ax, panel, (season, title) in zip(axs, "abc", SEASONS):
    for offset, key in ((-0.13, "M2"), (0.13, "M4")):
        d = gain[(gain.season == season) & (gain.forecast == key)].sort_values("lead")
        plot_interval(ax, d["lead"] + offset, d.skill.values, d.lo.values, d.hi.values, key, LABEL[key])
    format_lead_axis(ax)
    ax.set_title(f"({panel}) {title}", loc="left", fontsize=9)
axs[0].set_ylabel("MAE skill vs best\nsimple correction (%)")
legend_below(fig, axs[0], ncol=2, y=-0.02)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "Fig5_gain")

# ------------------------------------------------------------------ Figure 6: per province
province_gain = pd.read_csv(RESULTS_DIR / "province_gain.csv")
province_gain = province_gain[province_gain.forecast == "M4"]
limit = np.ceil(province_gain[province_gain["lead"].isin([1, 6])].skill.abs().max())
norm = TwoSlopeNorm(0, -limit, limit)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 6.0), gridspec_kw=dict(hspace=0.35))
for ax, panel, lead in zip(axs, "ab", (1, 6)):
    g = polygons.merge(province_gain[province_gain["lead"] == lead], on="province")
    g.plot(ax=ax, column="skill", cmap=DIVERGING, norm=norm, edgecolor="#ffffff", linewidth=0.4)
    significant = g[g.p_bh < 0.05]
    if len(significant):
        significant.plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=1.2, hatch="////")
    format_map_axes(ax)
    ax.set_title(f"({panel}) Lead {lead}: skill of M4 relative to EQM (B4)", loc="left", fontsize=9)
colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=DIVERGING, norm=norm), ax=axs, orientation="vertical", fraction=0.025, pad=0.02)
colorbar.set_label("MAE skill (%)")
save(fig, "Fig6_province", bbox_inches="tight")

# ------------------------------------------------------------------ Figure 7: RPSS
rpss = pd.read_csv(RESULTS_DIR / "rpss.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axs[0]
for key in ("B4", "M2", "M4"):
    d = rpss[(rpss.season == "all") & (rpss.forecast == key) & (rpss.reference == "clim")].sort_values("lead")
    ax.plot(d["lead"], d.rpss, color=COLOR[key], lw=2, marker=MARKER[key], ms=4, label=LABEL[key])
format_lead_axis(ax)
ax.set_ylabel("RPSS")
ax.set_title("(a) Relative to climatological probabilities", loc="left", fontsize=9)
ax = axs[1]
for offset, key in ((-0.13, "M2"), (0.13, "M4")):
    d = rpss[(rpss.season == "all") & (rpss.forecast == key) & (rpss.reference == "B4")].sort_values("lead")
    plot_interval(ax, d["lead"] + offset, d.rpss.values, d.lo.values, d.hi.values, key)
format_lead_axis(ax)
ax.set_title("(b) Relative to B4", loc="left", fontsize=9)
legend_below(fig, axs[0], ncol=3, y=-0.02)
fig.tight_layout(rect=(0, 0.09, 1, 1))
save(fig, "Fig7_rpss")

# ------------------------------------------------------------------ Figure 8: reliability
reliability = pd.read_csv(RESULTS_DIR / "reliability.csv")
fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.6), sharex=True, sharey=True)
for r, lead in enumerate((1, 3)):
    for c, category in enumerate(("BN", "AN")):
        ax = axs[r, c]
        ax.plot([0, 1], [0, 1], color=INK_2, lw=0.8, ls="--")
        ax.axhline(1 / 3, color="#c3c2b7", lw=0.8)
        for key, label in (("B4", "SEAS5, EQM members (B4)"), ("M4", "Gradient boosting (M4)")):
            d = reliability[(reliability["lead"] == lead) & (reliability.category == category) & (reliability.model == key) & (reliability.n >= 30)]
            ax.plot(d.p_mean, d.obs_freq, color=COLOR[key], lw=2, marker=MARKER[key], ms=4, label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"({'abcd'[2 * r + c]}) Lead {lead}, {'below' if category == 'BN' else 'above'} normal", loc="left", fontsize=9)
        if r == 1:
            ax.set_xlabel("Forecast probability")
        if c == 0:
            ax.set_ylabel("Observed frequency")
legend_below(fig, axs[0, 0], ncol=2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "Fig8_reliability")

# ------------------------------------------------------------------ Figure 9: ladder
ladder = pd.read_csv(RESULTS_DIR / "ladder.csv")
steps = [("M1", "B4", "(a) M1 vs B4: SEAS5 mean, spread, terciles"), ("M2", "M1", "(b) M2 vs M1: + recent observed rainfall"),
         ("M3", "M2", "(c) M3 vs M2: + neighbour SEAS5 anomaly"), ("M4", "M3", "(d) M4 vs M3: + Niño3.4 and DMI")]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=True)
for ax, (forecast, reference, title) in zip(axs.flat, steps):
    d = ladder[(ladder.season == "all") & (ladder.forecast == forecast) & (ladder.reference == reference)].sort_values("lead")
    ax.errorbar(d["lead"], d.skill, yerr=[d.skill - d.lo, d.hi - d.skill], fmt="o", color="#0d366b", ms=4.5, lw=1.6, capsize=0)
    format_lead_axis(ax)
    ax.set_title(title, loc="left", fontsize=9)
for ax in axs[:, 0]:
    ax.set_ylabel("MAE skill vs\nreference (%)")
fig.tight_layout()
save(fig, "Fig9_ladder")
print("figures:", sorted(p.name for p in FIGURES_DIR.glob("*.pdf")))
