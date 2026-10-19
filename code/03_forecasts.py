"""
Rolling-origin forecasts for the test years 2005-2024, leads 1-6: the simple references B0-B5 and the gradient
boosting models M1-M4, for rainfall amount (mm) and tercile probabilities (below, near and above normal).

Inputs
  data/seas5_province_members_long.parquet   (02_province_members.py)
  data/panel_rainfall_long.csv               (CHIRPS province rainfall, Nino3.4 and DMI, monthly)
  data/a_geo_centroids.csv                   (province centroids, for the neighbours)

Design (manuscript Section 3)
  Lead L, target month t: SEAS5 is initialised on the first day of month t-L+1; all other predictors use information
  up to the end of month t-L. For each test year T and lead L, every parameter is estimated from target months from
  January 1986 to month (January T) - L, the last month observed before the first forecast of year T is issued, and is
  applied to the 12 months of T.
  B0 climatology; B1 damped anomaly persistence; B2 raw SEAS5 ensemble mean; B3 mean-bias correction; B4 empirical
  quantile mapping (EQM) of each member; B5 linear regression of observed rainfall on the ensemble mean (MOS).
  B1, B3, B4 and B5 are fitted per province and calendar month.
  M1 LightGBM with SEAS5 mean, spread and tercile proportions, climatology, province and month; M2 adds the observed
  anomalies of months t-L, t-L-1 and t-L-2; M3 adds the mean SEAS5 anomaly of the k = 4 nearest provinces;
  M4 adds the 3-month means of Nino3.4 and DMI ending in month t-L. M1-M4 learn a correction to B3 (target y - B3);
  M1q-M4q learn a correction to B4 (target y - B4, with leave-one-year-out B4 in the training years).
  Checks: k = 2 and k = 6; M4d predicts y directly; M2q with settings tuned on the training years; M2q with a fixed
  number of trees (100 and 300); M2q with four other random seeds.
  Probabilities: B4 uses the proportion of members below/above the model terciles of the training years, shrunk
  towards 1/3 with a weight chosen by leave-one-year-out RPS; M1-M4 are three-class LightGBM classifiers.

Outputs (results/): forecasts_amount.parquet, forecasts_prob.parquet, n_trees.csv, neighbours_k4.csv, 03_forecasts.log
Usage: python code/03_forecasts.py [test years]    (with test years given, outputs get the suffix _subset)
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import FIRST_TRAIN_YEAR, LEADS, OBS_PANEL, PROVINCE_CENTROIDS, RESULTS_DIR, SEAS5_MEMBERS, TEST_YEARS, open_log
from scores import tercile_category

RESULTS_DIR.mkdir(exist_ok=True)
log = open_log(RESULTS_DIR / "03_forecasts.log")

test_years = [int(a) for a in sys.argv[1:]] or list(TEST_YEARS)
K_MAIN, K_CHECK = 4, (2, 6)                     # number of neighbours: main analysis and checks
# LightGBM settings fixed before the analysis and used for every lead, test year and model. Only the number of
# trees is chosen from the data: early stopping on the last 3 training years, then refit on all training years.
LGBM_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.9, reg_lambda=1.0, random_state=0, n_jobs=2, verbose=-1)
MAX_TREES, N_VALIDATION_YEARS, EARLY_STOPPING_ROUNDS = 1000, 3, 50
TUNING_GRID = [(lr, leaves, min_leaf) for lr in (0.03, 0.1) for leaves in (7, 15, 31) for min_leaf in (30, 100)]
SEEDS = (1, 2, 3, 4)                           # extra random seeds for the subsampling of M2q (main seed: 0)
FIXED_TREES = (100, 300)                       # M2q with a fixed number of trees (no early stopping)
CATEGORIES = ("BN", "NN", "AN")


def number_of_trees(task, X, y, years, params):
    """Early stopping on the last N_VALIDATION_YEARS training years. Returns the number of trees and the validation loss."""
    estimator = lgb.LGBMRegressor if task == "l1" else lgb.LGBMClassifier
    val = years >= years.max() - N_VALIDATION_YEARS + 1
    model = estimator(objective=task, n_estimators=MAX_TREES, **params).fit(
        X[~val], y[~val], eval_X=(X[val],), eval_y=(y[val],),
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return max(int(model.best_iteration_ or 1), 1), float(list(model.best_score_["valid_0"].values())[0])


def fit_lightgbm(task, X, y, years, **changes):
    """Fit a LightGBM regressor (task 'l1') or three-class classifier (task 'multiclass') on all training rows, with the
    number of trees chosen by early stopping. `changes` override entries of LGBM_PARAMS. Returns the model and the number of trees."""
    params = dict(LGBM_PARAMS, **changes)
    n_trees, _ = number_of_trees(task, X, y, years, params)
    estimator = lgb.LGBMRegressor if task == "l1" else lgb.LGBMClassifier
    return estimator(objective=task, n_estimators=n_trees, **params).fit(X, y), n_trees


def fit_lightgbm_tuned(X, y, years):
    """Regressor whose learning rate, number of leaves and minimum leaf size are chosen from TUNING_GRID by the validation
    error on the last N_VALIDATION_YEARS training years (the test year is never used). Returns model, trees and the setting."""
    results = []
    for lr, leaves, min_leaf in TUNING_GRID:
        params = dict(LGBM_PARAMS, learning_rate=lr, num_leaves=leaves, min_child_samples=min_leaf)
        n_trees, loss = number_of_trees("l1", X, y, years, params)
        results.append((loss, n_trees, params, f"lr{lr}_leaves{leaves}_min{min_leaf}"))
    _, n_trees, params, label = min(results, key=lambda r: r[0])
    return lgb.LGBMRegressor(objective="l1", n_estimators=n_trees, **params).fit(X, y), n_trees, label


def month_index(date):
    """Months since January 1985."""
    return (date.year - 1985) * 12 + date.month - 1


def three_month_mean(series, t):
    """Mean of months t-2, t-1 and t."""
    return np.nanmean(series[t - 2:t + 1])


def empirical_quantile_mapping(x, train_members, train_obs):
    """Map values x from the distribution of the training members to the distribution of the training observations.
    Values outside the training member range are shifted by the difference at the outermost quantile."""
    s = np.sort(train_members)
    n = len(s)
    prob = (np.searchsorted(s, x, "left") + np.searchsorted(s, x, "right")) / (2 * n)
    mapped = np.quantile(train_obs, np.clip(prob, 0, 1))
    shift_low = np.quantile(train_obs, 0.5 / n) - s[0]
    shift_high = np.quantile(train_obs, 1 - 0.5 / n) - s[-1]
    mapped = np.where(x < s[0], x + shift_low, np.where(x > s[-1], x + shift_high, mapped))
    return np.maximum(mapped, 0)


def shrinkage_weight(prob, category):
    """Weight w in [0, 1] that minimises the total RPS of w * prob + (1 - w) / 3 (closed form)."""
    clim_cum = np.array([1 / 3, 2 / 3])
    a = np.cumsum(prob, axis=1)[:, :2] - clim_cum
    b = clim_cum - np.stack([category <= 0, category <= 1], axis=1).astype(float)
    return float(np.clip(-(a * b).sum() / (a * a).sum(), 0, 1))


def member_tercile_proportions(x, lower, upper):
    return [(x < lower).mean(), ((x >= lower) & (x <= upper)).mean(), (x > upper).mean()]


def feature_matrix(D, model, k):
    """Predictors of model M1-M4; k is the number of neighbours used from M3 on."""
    X = pd.DataFrame({"province": D.province_idx.values, "month": D.month.values, "seas5_mean": D.seas5_mean.values,
                      "seas5_sd": D.seas5_sd.values, "climatology": D.clim.values,
                      "seas5_p_below": D.seas5_p_below.values, "seas5_p_above": D.seas5_p_above.values})
    if model >= 2:
        for j in range(3):
            X[f"obs_anom_lag{j}"] = D[f"obs_anom_lag{j}"].values
    if model >= 3:
        X["neighbour_anom"] = D[f"neighbour_anom_k{k}"].values
    if model >= 4:
        X["nino34"] = D.nino34.values
        X["dmi"] = D.dmi.values
    X["province"] = X.province.astype("category")
    X["month"] = X.month.astype("category")
    return X


# ------------------------------------------------------------------ observations and indices
obs = pd.read_csv(OBS_PANEL)
obs["date"] = pd.to_datetime(obs.year_month + "-01")
provinces = sorted(obs.province.unique())
n_prov = len(provinces)
assert n_prov == 38
n_months = month_index(obs.date.max()) + 1
obs_rain = np.full((n_prov, n_months), np.nan)
for i, province in enumerate(provinces):
    g = obs[obs.province == province]
    obs_rain[i, [month_index(d) for d in g.date]] = g.rainfall.values
indices = obs.drop_duplicates("date").set_index("date").sort_index()
nino34 = np.full(n_months, np.nan)
dmi = np.full(n_months, np.nan)
nino34[[month_index(d) for d in indices.index]] = indices.nino34.values
dmi[[month_index(d) for d in indices.index]] = indices.dmi.values

# ------------------------------------------------------------------ SEAS5 members: (province, initialisation, lead, member)
members = pd.read_parquet(SEAS5_MEMBERS)
members["province"] = members.province.astype(str)
province_idx = {p: i for i, p in enumerate(provinces)}
n_inits = 480                                                       # January 1985 to December 2024
seas5_members = np.full((n_prov, n_inits, 6, 51), np.nan, dtype="float32")
seas5_members[members.province.map(province_idx).values, members.init.map(month_index).values,
              members["lead"].values - 1, members.member.values] = members.rain_mm.values
del members
seas5_mean = np.nanmean(seas5_members, axis=3)
seas5_sd = np.nanstd(seas5_members, axis=3, ddof=1)
n_members = np.sum(~np.isnan(seas5_members), axis=3)
log("members per initialisation:", np.unique(n_members))

# ------------------------------------------------------------------ neighbours: nearest centroids (great-circle distance)
centroids = pd.read_csv(PROVINCE_CENTROIDS)
centroids = centroids[centroids.country == "IDN"].set_index("name").loc[provinces]
lat, lon = np.radians(centroids.lat.values), np.radians(centroids.lon.values)
distance = np.arccos(np.clip(np.sin(lat)[:, None] * np.sin(lat)[None] +
                             np.cos(lat)[:, None] * np.cos(lat)[None] * np.cos(lon[:, None] - lon[None]), -1, 1))
np.fill_diagonal(distance, np.inf)


def neighbour_weights(k):
    """Matrix (province x province) with weight 1/k on the k nearest provinces."""
    W = np.zeros((n_prov, n_prov))
    for i in range(n_prov):
        W[i, np.argsort(distance[i])[:k]] = 1 / k
    return W


NEIGHBOUR_W = {k: neighbour_weights(k) for k in (K_MAIN,) + K_CHECK}
pd.DataFrame([(provinces[i], provinces[j]) for i in range(n_prov) for j in np.where(NEIGHBOUR_W[K_MAIN][i] > 0)[0]],
             columns=["province", "neighbour"]).to_csv(RESULTS_DIR / "neighbours_k4.csv", index=False)

# ------------------------------------------------------------------ one row per (province, target month, lead), 1986-2024
rows = [(p, t, lead, t - lead + 1)
        for lead in LEADS
        for t in range(month_index(pd.Timestamp(f"{FIRST_TRAIN_YEAR}-01-01")), month_index(pd.Timestamp("2024-12-01")) + 1)
        for p in range(n_prov)]
panel = pd.DataFrame(rows, columns=["province_idx", "time_idx", "lead", "init_idx"])
panel["year"] = 1985 + panel.time_idx // 12
panel["month"] = panel.time_idx % 12 + 1
panel["y"] = obs_rain[panel.province_idx, panel.time_idx]
panel["seas5_mean"] = seas5_mean[panel.province_idx, panel.init_idx, panel["lead"] - 1]
panel["seas5_sd"] = seas5_sd[panel.province_idx, panel.init_idx, panel["lead"] - 1]
for j in range(3):                                                  # observed rainfall of months t-L, t-L-1, t-L-2
    lag_time = panel.time_idx - panel["lead"] - j
    panel[f"obs_lag{j}"] = obs_rain[panel.province_idx, lag_time]
    panel[f"month_lag{j}"] = lag_time % 12 + 1
panel["nino34"] = [three_month_mean(nino34, t - lead) for t, lead in zip(panel.time_idx, panel["lead"])]
panel["dmi"] = [three_month_mean(dmi, t - lead) for t, lead in zip(panel.time_idx, panel["lead"])]
assert panel[["y", "seas5_mean", "seas5_sd", "obs_lag0", "obs_lag1", "obs_lag2", "nino34", "dmi"]].notna().all().all()
log("rows:", len(panel), "| years:", panel.year.min(), "to", panel.year.max())

# ------------------------------------------------------------------ rolling-origin test
amount_parts, prob_parts, tree_counts = [], [], []
t_start = time.time()
for test_year in test_years:
    for lead in LEADS:
        # Training targets end in the month before the first forecast of the test year is issued: the forecast for January of
        # year T at lead L uses information up to the end of month (January T) - L.
        cutoff = month_index(pd.Timestamp(f"{test_year}-01-01")) - lead
        rows = panel[(panel["lead"] == lead) & (panel.year >= FIRST_TRAIN_YEAR)]
        in_train = rows.time_idx <= cutoff
        in_test = rows.year == test_year
        # observed climatology and tercile limits per province and calendar month, training targets only
        clim = rows[in_train].groupby(["province_idx", "month"]).y.mean()
        terciles = rows[in_train].groupby(["province_idx", "month"]).y.quantile([1 / 3, 2 / 3]).unstack()
        terciles.columns = ["lower", "upper"]
        D = rows[in_train | in_test].copy()
        D["clim"] = clim.reindex(pd.MultiIndex.from_arrays([D.province_idx, D.month])).values
        for j in range(3):
            D[f"obs_anom_lag{j}"] = D[f"obs_lag{j}"].values - clim.reindex(pd.MultiIndex.from_arrays([D.province_idx, D[f"month_lag{j}"]])).values
        D = D.join(terciles, on=["province_idx", "month"])
        D["obs_cat"] = tercile_category(D.y.values, D.lower.values, D.upper.values)
        # training mean of the SEAS5 ensemble mean per province and month (for B3 and the neighbour anomalies)
        is_train = D.time_idx <= cutoff
        seas5_clim = D[is_train].groupby(["province_idx", "month"]).seas5_mean.mean()
        D["seas5_clim"] = seas5_clim.reindex(pd.MultiIndex.from_arrays([D.province_idx, D.month])).values
        D["seas5_anom"] = D.seas5_mean - D.seas5_clim
        for k in (K_MAIN,) + K_CHECK:
            anom = D.pivot_table(index="time_idx", columns="province_idx", values="seas5_anom")
            nbr = pd.DataFrame(anom.values @ NEIGHBOUR_W[k].T, index=anom.index, columns=anom.columns).stack().rename(f"neighbour_anom_k{k}")
            D = D.join(nbr, on=["time_idx", "province_idx"])
        train = D[D.time_idx <= cutoff].copy()
        test = D[D.year == test_year].copy()
        train["seas5_p_below"] = np.nan
        train["seas5_p_above"] = np.nan
        train["B4_loo"] = np.nan

        # B1 damped anomaly persistence: correlation of the target anomaly with the anomaly of month t-L, clipped to [0, 1]
        damping = train.groupby(["province_idx", "month"])[["y", "clim", "obs_anom_lag0"]].apply(
            lambda g: np.clip(np.corrcoef(g.y - g.clim, g.obs_anom_lag0)[0, 1], 0, 1)).rename("damping")
        test = test.join(damping, on=["province_idx", "month"])
        test["B0"] = test.clim
        test["B1"] = test.clim + test.damping * test.obs_anom_lag0
        test["B2"] = test.seas5_mean
        test["B3"] = test.seas5_mean - (test.seas5_clim - test.clim)
        # B5 linear MOS
        mos = train.groupby(["province_idx", "month"])[["seas5_mean", "y"]].apply(
            lambda g: pd.Series(np.polyfit(g.seas5_mean, g.y, 1), index=["slope", "intercept"]))
        test = test.join(mos, on=["province_idx", "month"])
        test["B5"] = test.intercept + test.slope * test.seas5_mean

        # B4 EQM of each member and member tercile proportions; leave-one-year-out EQM forecasts and tercile proportions
        # in the training years (for the shrinkage weight and as the starting point of the gradient boosting models M1q-M4q)
        b4_amount, b4_prob = np.zeros(len(test)), np.zeros((len(test), 3))
        loo_prob, loo_cat = [], []
        for (p, month), g in test.groupby(["province_idx", "month"]):
            train_g = train[(train.province_idx == p) & (train.month == month)]
            train_members = seas5_members[p, train_g.init_idx.values, lead - 1]          # (training years, 51)
            pooled_members = train_members[~np.isnan(train_members)]
            train_y = train_g.y.values
            lower, upper = np.quantile(pooled_members, [1 / 3, 2 / 3])
            n_valid = np.sum(~np.isnan(train_members), axis=1)
            train.loc[train_g.index, "seas5_p_below"] = np.sum(train_members < lower, axis=1) / n_valid
            train.loc[train_g.index, "seas5_p_above"] = np.sum(train_members > upper, axis=1) / n_valid
            for row, init in zip(g.index, g.init_idx.values):
                x = seas5_members[p, init, lead - 1]
                x = x[~np.isnan(x)]
                pos = test.index.get_loc(row)
                b4_amount[pos] = empirical_quantile_mapping(x, pooled_members, train_y).mean()
                b4_prob[pos] = member_tercile_proportions(x, lower, upper)
            years = train_g.year.values
            b4_loo = np.zeros(len(years))
            for i, year in enumerate(years):
                keep = years != year
                kept = train_members[keep]
                kept = kept[~np.isnan(kept)]
                lower_loo, upper_loo = np.quantile(kept, [1 / 3, 2 / 3])
                x = train_members[i]
                x = x[~np.isnan(x)]
                loo_prob.append(member_tercile_proportions(x, lower_loo, upper_loo))
                lower_obs, upper_obs = np.quantile(train_y[keep], [1 / 3, 2 / 3])
                loo_cat.append(tercile_category(train_y[i], lower_obs, upper_obs))
                b4_loo[i] = max(empirical_quantile_mapping(x, kept, train_y[keep]).mean(), 0)
            train.loc[train_g.index, "B4_loo"] = b4_loo
        w_b4 = shrinkage_weight(np.array(loo_prob), np.array(loo_cat))
        test["B4"] = b4_amount
        test["seas5_p_below"] = b4_prob[:, 0]
        test["seas5_p_above"] = b4_prob[:, 2]
        assert train[["seas5_p_below", "seas5_p_above", "B4_loo"]].notna().all().all()
        b4_prob_shrunk = w_b4 * b4_prob + (1 - w_b4) / 3
        for c in ("B1", "B2", "B3", "B4", "B5"):
            test[c] = np.maximum(test[c], 0)

        # Gradient boosting learns a correction to a simple forecast: to B3 (models M1-M4) or to B4 (models M1q-M4q).
        # Checks: other numbers of neighbours (k), M4d (predicts y directly), M2q with tuned settings and with other seeds.
        base_train = {"": (train.seas5_mean - (train.seas5_clim - train.clim)).values, "q": train.B4_loo.values}
        base_test = {"": (test.seas5_mean - (test.seas5_clim - test.clim)).values, "q": test.B4.values}
        train_years = train.year.values
        for model in (1, 2, 3, 4):
            for k in (K_MAIN,) + (K_CHECK if model >= 3 else ()):
                X_train, X_test = feature_matrix(train, model, k), feature_matrix(test, model, k)
                for base in ("", "q"):
                    name = f"M{model}{base}" + ("" if k == K_MAIN else f"_k{k}")
                    reg, n_trees = fit_lightgbm("l1", X_train, train.y.values - base_train[base], train_years)
                    test[name] = np.maximum(base_test[base] + reg.predict(X_test), 0)
                    tree_counts.append((test_year, lead, name, "amount", n_trees))
                if model == 2 and k == K_MAIN:
                    reg, n_trees, params = fit_lightgbm_tuned(X_train, train.y.values - base_train["q"], train_years)
                    test["M2q_tuned"] = np.maximum(base_test["q"] + reg.predict(X_test), 0)
                    tree_counts.append((test_year, lead, "M2q_tuned " + params, "amount", n_trees))
                    for n_fixed in FIXED_TREES:
                        reg = lgb.LGBMRegressor(objective="l1", n_estimators=n_fixed, **LGBM_PARAMS).fit(X_train, train.y.values - base_train["q"])
                        test[f"M2q_trees{n_fixed}"] = np.maximum(base_test["q"] + reg.predict(X_test), 0)
                    for seed in SEEDS:
                        reg, n_trees = fit_lightgbm("l1", X_train, train.y.values - base_train["q"], train_years, random_state=seed)
                        test[f"M2q_seed{seed}"] = np.maximum(base_test["q"] + reg.predict(X_test), 0)
                if model == 4 and k == K_MAIN:
                    reg_direct, n_trees = fit_lightgbm("l1", X_train, train.y.values, train_years)
                    test["M4d"] = np.maximum(reg_direct.predict(X_test), 0)
                    tree_counts.append((test_year, lead, "M4d", "amount", n_trees))
                if k == K_MAIN:
                    clf, n_trees = fit_lightgbm("multiclass", X_train, train.obs_cat.values, train_years)
                    tree_counts.append((test_year, lead, f"M{model}", "prob", n_trees))
                    prob = clf.predict_proba(X_test)
                    for j, cat in enumerate(CATEGORIES):
                        test[f"M{model}_p{cat}"] = prob[:, j]
        for j, cat in enumerate(CATEGORIES):
            test[f"B4_p{cat}"] = b4_prob_shrunk[:, j]
        test["w_B4"] = w_b4
        test["n_mem"] = n_members[test.province_idx.values, test.init_idx.values, lead - 1]
        prob_cols = [c for c in test.columns if c.endswith(("_pBN", "_pNN", "_pAN"))]
        amount_cols = [c for c in test.columns if c[0] in "BM" and c[1].isdigit() and c not in prob_cols]
        id_cols = ["province_idx", "time_idx", "year", "month", "lead"]
        amount_parts.append(test[id_cols + ["y", "obs_cat", "lower", "upper", "n_mem"] + amount_cols].copy())
        prob_parts.append(test[id_cols + ["obs_cat", "w_B4"] + prob_cols])
    log(f"test year {test_year} done ({time.time() - t_start:.0f} s); B4 shrinkage weight by lead:",
        [round(float(x.w_B4.iloc[0]), 2) for x in prob_parts[-6:]])

amount = pd.concat(amount_parts, ignore_index=True)
prob = pd.concat(prob_parts, ignore_index=True)
amount["province"] = np.array(provinces)[amount.province_idx]
prob["province"] = np.array(provinces)[prob.province_idx]
suffix = "" if test_years == list(TEST_YEARS) else "_subset"
amount.to_parquet(RESULTS_DIR / f"forecasts_amount{suffix}.parquet", index=False)
prob.to_parquet(RESULTS_DIR / f"forecasts_prob{suffix}.parquet", index=False)
pd.DataFrame(tree_counts, columns=["test_year", "lead", "model", "target", "n_trees"]).to_csv(
    RESULTS_DIR / f"n_trees{suffix}.csv", index=False)
log("done:", len(amount), "amount rows,", len(prob), "probability rows; forecasts:", [c for c in amount.columns if c[0] in "BM"])
log.close()
