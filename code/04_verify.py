"""
Verification of the forecasts from 03_forecasts.py: MAE skill, Diebold-Mariano tests per province with the
Benjamini-Hochberg adjustment, year-bootstrap intervals, RPSS, reliability data, and the checks.

Inputs:  results/forecasts_amount.parquet, results/forecasts_prob.parquet
Outputs (results/):
  skill_amount.csv     MAE skill (%) relative to climatology, by forecast, lead and season, with 90% intervals
  annual_skill.csv     MAE skill (%) of B2, B4, M2, M4, M2q and M4q relative to climatology by test year and lead, and the
                       share (%) of months below and above normal in each test year
  best_simple.csv      most accurate simple correction (B3, B4 or B5) for each lead and season
  gain_vs_simple.csv   MAE skill (%) of the gradient boosting models, the tuned, fixed-tree and other-seed versions of M2q and the
                       equal-weight combinations relative to that correction, with the number of provinces significantly
                       better or worse (Diebold-Mariano with HLN correction, then Benjamini-Hochberg)
  ladder.csv           MAE skill (%) of each model relative to the one below it, for both starting points (B3 and B4)
  rpss.csv             RPSS relative to climatological probabilities and to B4, with 90% intervals
  reliability.csv      observed frequency in 10 probability bins, below and above normal, B4 and M4, all leads
  province_gain.csv    MAE skill per province and lead of M2, M4, M2q, M4q and B4+M2q relative to the best simple correction
  checks.csv           variants (k = 2 and 6, M4d, M1 and M2 relative to B3), averages compared with each other, and the
                       periods 2005-2016 and 2017-2024
  04_verify.log
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import LEADS, RESULTS_DIR, TEST_YEARS, open_log
from scores import ranked_probability_score

log = open_log(RESULTS_DIR / "04_verify.log")
amount = pd.read_parquet(RESULTS_DIR / "forecasts_amount.parquet")
prob = pd.read_parquet(RESULTS_DIR / "forecasts_prob.parquet")

N_BOOT = 2000
YEARS = np.array(TEST_YEARS)
SEASONS = {"all": lambda m: np.ones(len(m), bool),
           "JunNov": lambda m: (m >= 6) & (m <= 11),
           "DecMay": lambda m: (m <= 5) | (m == 12)}
# one set of resampled years shared by all comparisons (paired bootstrap)
BOOT_INDEX = np.random.default_rng(42).integers(0, len(YEARS), (N_BOOT, len(YEARS)))


def sum_by_year(values, years):
    return np.bincount(np.searchsorted(YEARS, years), weights=values, minlength=len(YEARS))


def skill_score(loss_forecast, loss_reference, years):
    """Skill (%) = 100 * (1 - sum of forecast losses / sum of reference losses), with a 90% interval from
    resampling whole test years."""
    a, b = sum_by_year(loss_forecast, years), sum_by_year(loss_reference, years)
    skill = 1 - a.sum() / b.sum()
    boot = 1 - a[BOOT_INDEX].sum(axis=1) / b[BOOT_INDEX].sum(axis=1)
    return 100 * skill, 100 * np.percentile(boot, 5), 100 * np.percentile(boot, 95)


def diebold_mariano_hln(e1, e2, horizon):
    """Diebold-Mariano test on the difference in absolute errors, Newey-West variance (Bartlett kernel, bandwidth
    max(4, horizon - 1)), with the Harvey-Leybourne-Newbold small-sample correction. Returns the statistic and the
    two-sided p-value."""
    bandwidth = max(4, horizon - 1)
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n + 2 * sum((1 - k / (bandwidth + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, bandwidth + 1))
    if var <= 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(var / n) * np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), n - 1)


def benjamini_hochberg(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(q, 1)
    return adjusted


def province_tests(D, forecast, reference, horizon):
    rows = []
    for province, g in D.groupby("province"):
        g = g.sort_values("time_idx")
        skill = 100 * (1 - np.abs(g.y - g[forecast]).sum() / np.abs(g.y - g[reference]).sum())
        stat, p = diebold_mariano_hln((g.y - g[forecast]).values, (g.y - g[reference]).values, horizon)
        rows.append((province, skill, stat, p))
    out = pd.DataFrame(rows, columns=["province", "skill", "dm", "p"])
    out["p_bh"] = benjamini_hochberg(out.p)
    return out


def compare(D, forecast, reference, lead, season):
    """MAE skill of `forecast` relative to `reference` over all provinces, and the per-province tests."""
    D = D[SEASONS[season](D.month.values)]
    skill, lo, hi = skill_score(np.abs(D.y - D[forecast]).values, np.abs(D.y - D[reference]).values, D.year.values)
    per_province = province_tests(D, forecast, reference, lead)
    significant = per_province.p_bh < 0.05
    summary = dict(lead=lead, season=season, forecast=forecast, reference=reference, skill=skill, lo=lo, hi=hi,
                   n_better=int((significant & (per_province.skill > 0)).sum()),
                   n_worse=int((significant & (per_province.skill < 0)).sum()))
    return summary, per_province


# ------------------------------------------------------------------ forecast combinations (equal weights, nothing fitted)
COMBINATIONS = {"C_B4_B5": ("B4", "B5"), "C_B4_M1": ("B4", "M1"), "C_B4_M2": ("B4", "M2"), "C_B4_M2q": ("B4", "M2q")}
for name, (f1, f2) in COMBINATIONS.items():
    amount[name] = (amount[f1] + amount[f2]) / 2

# ------------------------------------------------------------------ MAE skill relative to climatology
SIMPLE = ("B3", "B4", "B5")
MODELS_B3 = ("M1", "M2", "M3", "M4")
MODELS_B4 = ("M1q", "M2q", "M3q", "M4q")
SENSITIVITY = ("M2q_tuned", "M2q_trees100", "M2q_trees300", "M2q_seed1", "M2q_seed2", "M2q_seed3", "M2q_seed4")
VARIANTS = ("M4d", "M3_k2", "M4_k2", "M3_k6", "M4_k6", "M3q_k2", "M4q_k2", "M3q_k6", "M4q_k6")
FORECASTS = ["B1", "B2", *SIMPLE, *MODELS_B3, *MODELS_B4, *SENSITIVITY, *VARIANTS, *COMBINATIONS]
rows = []
for lead in LEADS:
    D = amount[amount["lead"] == lead]
    for season, in_season in SEASONS.items():
        Ds = D[in_season(D.month.values)]
        for f in FORECASTS:
            skill, lo, hi = skill_score(np.abs(Ds.y - Ds[f]).values, np.abs(Ds.y - Ds.B0).values, Ds.year.values)
            rows.append(dict(lead=lead, season=season, method=f, skill=skill, lo=lo, hi=hi, mae=np.abs(Ds.y - Ds[f]).mean()))
skill_table = pd.DataFrame(rows)
skill_table.to_csv(RESULTS_DIR / "skill_amount.csv", index=False)
log("MAE skill (%) relative to climatology, all months:\n"
    + skill_table[skill_table.season == "all"].pivot(index="method", columns="lead", values="skill").round(1).to_string())

# MAE skill relative to climatology for each test year, and the share of months in each observed category
rows = []
for (lead, year), D in amount.groupby(["lead", "year"]):
    mae_clim = np.abs(D.y - D.B0).sum()
    row = dict(lead=lead, year=year, pct_below=100 * (D.obs_cat == 0).mean(), pct_above=100 * (D.obs_cat == 2).mean())
    row.update({f: 100 * (1 - np.abs(D.y - D[f]).sum() / mae_clim) for f in ("B2", "B4", "M2", "M4", "M2q", "M4q")})
    rows.append(row)
annual = pd.DataFrame(rows)
annual.to_csv(RESULTS_DIR / "annual_skill.csv", index=False)
log("\nMAE skill (%) of B4 relative to climatology by test year:\n"
    + annual.pivot(index="year", columns="lead", values="B4").round(1).to_string())

# Most accurate simple correction (B3, B4 or B5) for each lead and season, chosen on the test data. Choosing the strongest
# reference after the test can only favour the reference.
mae = skill_table.set_index(["lead", "season", "method"]).mae
best_simple = {(lead, season): min(SIMPLE, key=lambda c: mae[(lead, season, c)]) for lead in LEADS for season in SEASONS}
pd.Series(best_simple, name="best_simple").rename_axis(["lead", "season"]).to_csv(RESULTS_DIR / "best_simple.csv")
log("most accurate simple correction by lead and season:", best_simple)

gain_rows, ladder_rows, province_rows, check_rows = [], [], [], []
for lead in LEADS:
    D = amount[amount["lead"] == lead]
    for season in SEASONS:
        best = best_simple[(lead, season)]
        for f in (*MODELS_B3, *MODELS_B4, *SENSITIVITY, *COMBINATIONS):
            summary, per_province = compare(D, f, best, lead, season)
            gain_rows.append(summary)
            if season == "all" and f in ("M2", "M4", "M2q", "M4q", "C_B4_M2q"):
                province_rows.append(per_province.assign(lead=lead, forecast=f, reference=best))
        for models in (MODELS_B3, MODELS_B4):
            for f, ref in zip(models, (best,) + models[:-1]):
                ladder_rows.append(compare(D, f, ref, lead, season)[0])
        check_rows.append(dict(compare(D, "C_B4_M2", "C_B4_B5", lead, season)[0], check="combination"))
        check_rows.append(dict(compare(D, "C_B4_M1", "C_B4_B5", lead, season)[0], check="combination"))
        if season == "all":
            for f, ref in (("M3_k2", "M3"), ("M3_k6", "M3"), ("M4_k2", "M4"), ("M4_k6", "M4"), ("M3q_k2", "M3q"), ("M3q_k6", "M3q"),
                           ("M4q_k2", "M4q"), ("M4q_k6", "M4q"), ("M4d", "M4"), ("M4d", best), ("M1", "B3"), ("M2", "B3"),
                           ("C_B4_M2", "C_B4_M1")):
                check_rows.append(dict(compare(D, f, ref, lead, season)[0], check="variant"))
    best = best_simple[(lead, "all")]
    for period, (y0, y1) in {"2005-2016": (2005, 2016), "2017-2024": (2017, 2024)}.items():
        Dp = D[D.year.between(y0, y1)]
        for f, ref in (("B4", "B0"), ("B3", "B0"), ("M4", "B0"), ("M4", best), ("M2q", best)):
            skill = 100 * (1 - np.abs(Dp.y - Dp[f]).sum() / np.abs(Dp.y - Dp[ref]).sum())
            check_rows.append(dict(lead=lead, season="all", forecast=f, reference=ref, skill=skill, check=period))
gain = pd.DataFrame(gain_rows)
ladder = pd.DataFrame(ladder_rows)
gain.to_csv(RESULTS_DIR / "gain_vs_simple.csv", index=False)
ladder.to_csv(RESULTS_DIR / "ladder.csv", index=False)
pd.concat(province_rows).to_csv(RESULTS_DIR / "province_gain.csv", index=False)
pd.DataFrame(check_rows).to_csv(RESULTS_DIR / "checks.csv", index=False)
for season in SEASONS:
    log(f"\nMAE skill (%) relative to the best simple correction, season {season}:\n"
        + gain[gain.season == season].pivot(index="forecast", columns="lead", values="skill").round(1).to_string())
log("\nladder (all months):\n" + ladder[ladder.season == "all"][
    ["lead", "forecast", "reference", "skill", "lo", "hi", "n_better", "n_worse"]].round(1).to_string(index=False))

# ------------------------------------------------------------------ tercile probabilities
rpss_rows, reliability_rows = [], []
MODELS = ("B4", "M1", "M2", "M3", "M4")
for lead in LEADS:
    D = prob[prob["lead"] == lead]
    category = D.obs_cat.values.astype(int)
    probs = {m: D[[f"{m}_pBN", f"{m}_pNN", f"{m}_pAN"]].values for m in MODELS}
    probs["clim"] = np.full((len(D), 3), 1 / 3)
    for season, in_season in SEASONS.items():
        k = in_season(D.month.values)
        for m in MODELS:
            for ref in ("clim", "B4"):
                if m == ref:
                    continue
                skill, lo, hi = skill_score(ranked_probability_score(probs[m][k], category[k]),
                                            ranked_probability_score(probs[ref][k], category[k]), D.year.values[k])
                rpss_rows.append(dict(lead=lead, season=season, forecast=m, reference=ref, rpss=skill / 100, lo=lo / 100, hi=hi / 100))
    for m in ("B4", "M4"):
        for j, name in ((0, "BN"), (2, "AN")):
            p = probs[m][:, j]
            event = (category == j).astype(float)
            bins = np.minimum((p * 10).astype(int), 9)
            for b in range(10):
                in_bin = bins == b
                if in_bin.sum():
                    reliability_rows.append(dict(lead=lead, model=m, category=name, bin=b, p_mean=p[in_bin].mean(),
                                                 obs_freq=event[in_bin].mean(), n=int(in_bin.sum())))
rpss = pd.DataFrame(rpss_rows)
rpss.to_csv(RESULTS_DIR / "rpss.csv", index=False)
pd.DataFrame(reliability_rows).to_csv(RESULTS_DIR / "reliability.csv", index=False)
log("\nRPSS relative to climatology, all months:\n"
    + rpss[(rpss.season == "all") & (rpss.reference == "clim")].pivot(index="forecast", columns="lead", values="rpss").round(3).to_string())
log("\nRPSS relative to B4, all months:\n"
    + rpss[(rpss.season == "all") & (rpss.reference == "B4")].pivot(index="forecast", columns="lead", values="rpss").round(3).to_string())
log.close()
