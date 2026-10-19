"""
LaTeX tables of the manuscript (manuscript/tables/*.tex), the supplementary tables (manuscript/supplementary_tables.xlsx,
also as results/supplementary_S*.csv) and a text summary of the numbers quoted in the text (results/numbers.txt), from the
outputs of 03_forecasts.py and 04_verify.py.
The .tex files are generated: change this script and run it again rather than editing them.
Usage: python code/06_tables.py
"""
import numpy as np
import pandas as pd

from config import FIRST_TRAIN_YEAR, LEADS, OBS_PANEL, RESULTS_DIR, SEAS5_ENSEMBLE, TABLES_DIR

TABLES_DIR.mkdir(parents=True, exist_ok=True)
skill = pd.read_csv(RESULTS_DIR / "skill_amount.csv")
gain = pd.read_csv(RESULTS_DIR / "gain_vs_simple.csv")
ladder = pd.read_csv(RESULTS_DIR / "ladder.csv")
rpss = pd.read_csv(RESULTS_DIR / "rpss.csv")
checks = pd.read_csv(RESULTS_DIR / "checks.csv")
n_trees = pd.read_csv(RESULTS_DIR / "n_trees.csv")
amount = pd.read_parquet(RESULTS_DIR / "forecasts_amount.parquet")


def fmt(x, decimals=1):
    """Number with a typographic minus sign; -0.0 is written as 0.0."""
    s = f"{x:.{decimals}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "\u2212")


def select(table, **conditions):
    mask = np.ones(len(table), bool)
    for column, value in conditions.items():
        mask &= (table[column] == value).values
    return table[mask]


# ------------------------------------------------------------------ Table: MAE skill relative to climatology
names = [("B1", "Damped anomaly persistence"), ("B2", "Raw SEAS5"), ("B3", "SEAS5, mean-bias correction"), ("B4", "SEAS5, EQM"),
         ("B5", "SEAS5, linear MOS"), ("M1", "Gradient boosting M1"), ("M2", "Gradient boosting M2"), ("M3", "Gradient boosting M3"),
         ("M4", "Gradient boosting M4")]
rows = []
for method, label in names:
    d = select(skill, season="all", method=method).set_index("lead")
    rows.append(f"{method} & {label} & " + " & ".join(fmt(d.loc[lead, "skill"]) for lead in LEADS) + r" \\")
b4 = select(skill, season="all", method="B4").set_index("lead")
table = (r"\begin{table}[!htbp]\centering\small" "\n"
         r"\caption{MAE skill (\%) relative to climatology (B0), all months, 2005--2024.}\label{tab:skill}" "\n"
         r"\begin{tabular}{@{}llrrrrrr@{}}" "\n" r"\toprule" "\n"
         r" & & \multicolumn{6}{c}{Lead time (months)} \\ \cmidrule(l){3-8}" "\n"
         r"Label & Forecast & 1 & 2 & 3 & 4 & 5 & 6 \\" "\n" r"\midrule" "\n"
         + "\n".join(rows[:5]) + "\n" r"\midrule" "\n" + "\n".join(rows[5:]) + "\n"
         r"\bottomrule" "\n" r"\end{tabular}" "\n\n"
         r"\smallskip\parbox{\linewidth}{\footnotesize 90\% year-bootstrap intervals for B4: "
         + "; ".join(f"lead {lead} [{fmt(b4.loc[lead, 'lo'])}, {fmt(b4.loc[lead, 'hi'])}]" for lead in LEADS) + ".}\n"
         r"\end{table}" "\n")
(TABLES_DIR / "table_skill.tex").write_text(table, encoding="utf-8")

# ------------------------------------------------------------------ Table: robustness of the comparison with B4
def cell(row):
    """Skill with an asterisk when the 90% interval excludes zero."""
    star = row.lo > 0 or row.hi < 0
    return fmt(row.skill) + (r"\rlap{$^{*}$}" if star else "")


def table_row(forecast, label):
    d = select(gain, season="all", forecast=forecast).set_index("lead")
    return f"{label} & " + " & ".join(cell(d.loc[lead]) for lead in LEADS) + r" \\"


seeds = gain[(gain.season == "all") & gain.forecast.isin(["M2q"] + [f"M2q_seed{i}" for i in range(1, 5)])]
spread = seeds.groupby("lead").skill.agg(lambda x: x.max() - x.min())
rows = [table_row("M2", "M2 (from B3)"), table_row("M4", "M4 (from B3)"), table_row("M2q", "M2q (from B4)"), table_row("M4q", "M4q (from B4)"),
        table_row("M2q_tuned", "M2q, tuned settings"), table_row("M2q_trees100", "M2q, 100 trees"), table_row("M2q_trees300", "M2q, 300 trees"),
        r"\midrule",
        table_row("C_B4_B5", "Average of B4 and B5"), table_row("C_B4_M1", "Average of B4 and M1"), table_row("C_B4_M2", "Average of B4 and M2"),
        table_row("C_B4_M2q", "Average of B4 and M2q")]
table = (r"\begin{table}[!htbp]\centering\small" "\n"
         r"\caption{MAE skill (\%) relative to SEAS5 with EQM (B4), all months, 2005--2024. Positive values: more accurate than B4.}\label{tab:robust}" "\n"
         r"\begin{tabular}{@{}lrrrrrr@{}}" "\n" r"\toprule" "\n"
         r" & \multicolumn{6}{c}{Lead time (months)} \\ \cmidrule(l){2-7}" "\n"
         r"Forecast & 1 & 2 & 3 & 4 & 5 & 6 \\" "\n" r"\midrule" "\n" + "\n".join(rows) + "\n"
         r"\bottomrule" "\n" r"\end{tabular}" "\n\n"
         r"\smallskip\parbox{\linewidth}{\footnotesize $^{*}$ The 90\% year-bootstrap interval excludes zero. ``From B3'' and ``from B4'': the "
         r"starting point whose error the model learns. The tuned version chooses its settings on the training years; the 100- and 300-tree "
         r"versions use no early stopping. With five random seeds, the skill of M2q varied by at most "
         + fmt(spread.max(), 1) + r" percentage points. The averages give equal weight to the two forecasts and fit nothing.}" "\n"
         r"\end{table}" "\n")
(TABLES_DIR / "table_robust.tex").write_text(table, encoding="utf-8")

# ------------------------------------------------------------------ supplementary tables (Excel file, also as CSV)
LABEL = {"M1": "M1", "M2": "M2", "M3": "M3", "M4": "M4", "M1q": "M1q", "M2q": "M2q", "M3q": "M3q", "M4q": "M4q",
         "M2q_tuned": "M2q tuned", "M2q_trees100": "M2q 100 trees", "M2q_trees300": "M2q 300 trees",
         "C_B4_B5": "average of B4 and B5", "C_B4_M1": "average of B4 and M1", "C_B4_M2": "average of B4 and M2",
         "C_B4_M2q": "average of B4 and M2q", "B0": "B0", "B1": "B1", "B2": "B2", "B3": "B3", "B4": "B4", "B5": "B5", "clim": "climatology",
         "M4d": "M4d", "M3_k2": "M3 (k = 2)", "M3_k6": "M3 (k = 6)", "M4_k2": "M4 (k = 2)", "M4_k6": "M4 (k = 6)", "M3q_k2": "M3q (k = 2)",
         "M3q_k6": "M3q (k = 6)", "M4q_k2": "M4q (k = 2)", "M4q_k6": "M4q (k = 6)"}
SEASON_NAME = {"all": "all months", "JunNov": "June-November", "DecMay": "December-May"}


def readable(frame, decimals):
    out = frame.copy()
    for col in ("forecast", "reference"):
        out[col] = out[col].map(lambda x: LABEL.get(x, x))
    out["season"] = out["season"].map(SEASON_NAME)
    return out.round(decimals)


models = [m for m in LABEL if m.startswith(("M", "C_")) and m in set(gain.forecast)]
sheets = {
    "S1": readable(gain[gain.forecast.isin(models)][["lead", "season", "forecast", "reference", "skill", "lo", "hi", "n_better", "n_worse"]], 2),
    "S2": readable(rpss[["lead", "season", "forecast", "reference", "rpss", "lo", "hi"]], 3),
    "S3": readable(ladder[["lead", "season", "forecast", "reference", "skill", "lo", "hi", "n_better", "n_worse"]], 2),
    "S4": readable(checks[["check", "lead", "season", "forecast", "reference", "skill", "lo", "hi", "n_better", "n_worse"]], 2),
}
readme = pd.DataFrame({"item": [
    "S1", "S2", "S3", "S4", "skill", "rpss", "lo, hi", "n_better, n_worse", "B0-B5", "M1-M4", "M1q-M4q", "average of X and Y", "season"],
    "description": [
    "MAE skill (%) of the gradient boosting models and of equal-weight averages relative to the most accurate simple correction "
    "(B3, B4 or B5) for each lead and season",
    "Ranked probability skill score of tercile probabilities relative to climatological probabilities and to B4",
    "MAE skill (%) of each gradient boosting model relative to the model before it (M1 and M1q: relative to the most accurate "
    "simple correction)",
    "Checks: other numbers of neighbours (k), M4d, M1 and M2 relative to B3, averages compared with each other, and the periods "
    "2005-2016 and 2017-2024 (no intervals for the periods)",
    "MAE skill in % of 'forecast' relative to 'reference': 100 * (1 - MAE_forecast / MAE_reference)",
    "1 - RPS_forecast / RPS_reference",
    "90% interval from a paired bootstrap over test years (2000 replicates)",
    "number of the 38 provinces in which the forecast is significantly more / less accurate than the reference "
    "(Diebold-Mariano test, Benjamini-Hochberg adjustment at a false discovery rate of 0.05)",
    "B0 climatology, B1 damped anomaly persistence, B2 raw SEAS5, B3 mean-bias correction, B4 empirical quantile mapping, "
    "B5 linear MOS",
    "LightGBM learning a correction to B3; M1 SEAS5 predictors, M2 + recent observed rainfall, M3 + neighbour SEAS5 anomaly, "
    "M4 + Nino3.4 and DMI",
    "as M1-M4 but learning a correction to B4; 'tuned': settings chosen on the training years; '100/300 trees': no early stopping",
    "equal-weight average of the two forecasts",
    "all months, June-November or December-May"]})
with pd.ExcelWriter(TABLES_DIR.parent / "supplementary_tables.xlsx") as writer:
    readme.to_excel(writer, sheet_name="README", index=False)
    for name, frame in sheets.items():
        frame.to_excel(writer, sheet_name=name, index=False)
for name, frame in sheets.items():
    frame.to_csv(RESULTS_DIR / f"supplementary_{name}.csv", index=False)

# ------------------------------------------------------------------ numbers quoted in the text
out = []
for season in ("all", "JunNov", "DecMay"):
    out.append(f"\n== MAE skill (%) relative to climatology, season {season}")
    out.append(skill[skill.season == season].pivot(index="method", columns="lead", values="skill").round(1).to_string())
out.append("\n== MAE skill with 90% intervals, all months")
out.append(skill[(skill.season == "all") & skill.method.isin(["B1", "B2", "B3", "B4", "B5", "M2", "M4"])]
           [["method", "lead", "skill", "lo", "hi"]].round(1).to_string(index=False))
out.append("\n== M1-M4 relative to the best simple correction")
out.append(gain.round(1).to_string(index=False))
out.append("\n== ladder")
out.append(ladder.round(1).to_string(index=False))
out.append("\n== RPSS")
out.append(rpss.round(3).to_string(index=False))
out.append("\n== checks")
out.append(checks.round(1).to_string(index=False))
out.append("\n== number of trees (median, min, max)")
out.append(n_trees.groupby(["target", "model"]).n_trees.agg(["median", "min", "max"]).to_string())
all_months = skill[skill.season == "all"].pivot(index="method", columns="lead", values="skill")
out.append("\n== skill of B4 minus skill of B3, B5 and B1 (percentage points)")
out.append((all_months.loc[["B4"]].values - all_months.loc[["B3", "B5", "B1"]]).round(1).to_string())
lead1 = amount[amount["lead"] == 1]
out.append(f"\nrows: {len(amount)}; MAE of climatology, all months (mm): {np.abs(lead1.y - lead1.B0).mean():.1f}")
out.append("share of test months in each category of the training-year terciles (lead 1): "
           + ", ".join(f"{name} {100 * (lead1.obs_cat == j).mean():.0f}%" for j, name in enumerate(("below", "near", "above"))))

# correlation between the SEAS5 anomaly of a province and the mean anomaly of its four neighbours, lead 1, 1986-2024
ensemble = pd.read_parquet(SEAS5_ENSEMBLE)
ensemble = ensemble[(ensemble["lead"] == 1) & ensemble.date.dt.year.between(FIRST_TRAIN_YEAR, 2024)].copy()
ensemble["province"] = ensemble.province.astype(str)
ensemble["anom"] = ensemble.ens_mean - ensemble.groupby(["province", ensemble.date.dt.month]).ens_mean.transform("mean")
anom = ensemble.pivot(index="date", columns="province", values="anom")
neighbours = pd.read_csv(RESULTS_DIR / "neighbours_k4.csv").groupby("province").neighbour.apply(list)
corr = pd.Series({p: np.corrcoef(anom[p], anom[nbrs].mean(axis=1))[0, 1] for p, nbrs in neighbours.items()})
out.append(f"correlation of province and neighbour-mean SEAS5 anomaly, lead 1: median {corr.median():.2f}, "
           f"range {corr.min():.2f} to {corr.max():.2f}")
# skill in each test year
annual = pd.read_csv(RESULTS_DIR / "annual_skill.csv")
for lead in (1, 3):
    d = annual[annual["lead"] == lead].set_index("year")
    out.append(f"\n== lead {lead}, skill by test year: B4 from {d.B4.min():.1f} ({d.B4.idxmin()}) to {d.B4.max():.1f} ({d.B4.idxmax()}); "
               f"years with B4 below zero: {list(d.index[d.B4 < 0])}")
    for model in ("M2", "M4", "M2q"):
        diff = d[model] - d.B4
        out.append(f"   {model} - B4: from {diff.min():.1f} ({diff.idxmin()}) to {diff.max():.1f} ({diff.idxmax()})")
out.append(annual.round(1).to_string(index=False))

# province-level tests of M4 relative to B4: raw p-values below 0.05 and provinces significant after BH
province_gain = pd.read_csv(RESULTS_DIR / "province_gain.csv")
for f in ("M2", "M4", "M2q", "M4q"):
    d = province_gain[province_gain.forecast == f]
    raw = d[d.p < 0.05].groupby("lead").apply(lambda g: f"{(g.skill > 0).sum()} better / {(g.skill < 0).sum()} worse", include_groups=False)
    bh = d[d.p_bh < 0.05][["lead", "province", "skill"]].round(1).values.tolist()
    out.append(f"\n== {f} vs best simple, provinces with raw p < 0.05 by lead: {raw.to_dict()}; significant after BH: {bh}")

# wetter test period, and B4 relative to a climatology of the 10 most recent training years (lead 1)
obs = pd.read_csv(OBS_PANEL)
obs["year"] = obs.year_month.str[:4].astype(int)
annual_rain = obs.groupby(["province", "year"]).rainfall.sum().groupby("year").mean()
train_mean, test_mean = annual_rain.loc[1986:2004].mean(), annual_rain.loc[2005:2024].mean()
out.append(f"\nmean annual province rainfall: 1986-2004 {train_mean:.0f} mm, 2005-2024 {test_mean:.0f} mm "
           f"({100 * (test_mean / train_mean - 1):.0f}% more)")
obs["month"] = obs.year_month.str[5:].astype(int)
panel_obs = obs.set_index(["province", "year", "month"]).rainfall
lead1 = lead1.assign(province=np.array(sorted(obs.province.unique()))[lead1.province_idx])
recent = []
for (prov, year, month) in zip(lead1.province, lead1.year, lead1.month):
    recent.append(np.mean([panel_obs[(prov, y, month)] for y in range(year - 10, year)]))
lead1 = lead1.assign(clim10=recent)
mae = lambda f: np.abs(lead1.y - lead1[f]).sum()
out.append(f"lead 1: climatology of the last 10 training years vs B0: {100 * (1 - mae('clim10') / mae('B0')):.1f}%; "
           f"B4 vs 10-year climatology: {100 * (1 - mae('B4') / mae('clim10')):.1f}%")
# June-November means of Nino3.4 and DMI in each test year (context for Fig. 4)
indices = obs.drop_duplicates("year_month")
indices = indices[indices.month.between(6, 11) & indices.year.between(2005, 2024)].groupby("year")[["nino34", "dmi"]].mean()
out.append("\n== June-November mean Nino3.4 and DMI (degC) by test year\n" + indices.round(2).to_string())
out.append(f"training rows per gradient boosting fit: {38 * (19 * 12 - 5)} (2005, lead 6) to {38 * 38 * 12} (2024, lead 1)")
# B4 skill per province, June-November, lead 1
jn = amount[(amount["lead"] == 1) & amount.month.between(6, 11)].assign(province=lambda d: np.array(sorted(obs.province.unique()))[d.province_idx])
prov_skill = jn.groupby("province").apply(lambda g: 100 * (1 - np.abs(g.y - g.B4).sum() / np.abs(g.y - g.B0).sum()), include_groups=False)
out.append("\n== B4 skill (%) relative to climatology per province, June-November, lead 1\n" + prov_skill.sort_values().round(1).to_string())
(RESULTS_DIR / "numbers.txt").write_text("\n".join(out), encoding="utf-8")
print("tables:", sorted(p.name for p in TABLES_DIR.glob("*.tex")))
