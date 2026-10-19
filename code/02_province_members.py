"""
SEAS5 rainfall (mm per month) averaged over each of the 38 provinces, for every ensemble member, leads 1-6,
initialisations 1985-2024.

Inputs
  data/c3s_seas5_lead1to6/seas5_tprate_lead1-6_YYYY.grib     (01_download_seas5.py)
  data/provinces_simplified.geojson                           (province polygons)
  data/panel_rainfall_long.csv                                (CHIRPS province rainfall; used only for the bias summary in the log)

Method
  Each 1 degree grid cell is weighted by the area of its intersection with the province polygon (equal-area projection
  EPSG:6933); the weights of a province sum to one, so small provinces such as Jakarta also receive a value.
  Precipitation rate (m/s) is converted to mm per month by multiplying by 1000 and by the number of seconds in the
  target month. At lead L the target month is the initialisation month + (L - 1).

Outputs (data/)
  seas5_province_members_long.parquet   province, init, lead, date, member, rain_mm   (one row per member)
  seas5_province_ensemble.parquet       province, init, lead, date, n_mem, ens_mean, ens_sd
  02_province_members.log               row counts, member counts, and mean bias against CHIRPS by lead

Usage (from the project folder):  python code/02_province_members.py
"""
import calendar
import os
import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, GRIB_DIR, INIT_YEARS, OBS_PANEL, PROVINCE_NAME_EN, PROVINCE_POLYGONS, SEAS5_ENSEMBLE, SEAS5_MEMBERS, open_log

GRIB_OPTIONS = dict(engine="cfgrib", backend_kwargs=dict(time_dims=("forecastMonth", "time"), indexpath=""))


def province_area_weights(lat, lon):
    """Weight matrix (38 provinces x grid cells) from cell-polygon intersection areas; each row sums to one."""
    import geopandas as gpd
    from shapely.geometry import box
    polygons = gpd.read_file(PROVINCE_POLYGONS)
    polygons["province"] = polygons.PROVINSI.map(PROVINCE_NAME_EN)
    assert polygons.province.notna().all() and polygons.province.nunique() == 38, "province names do not match"
    dlat, dlon = abs(float(lat[1] - lat[0])), abs(float(lon[1] - lon[0]))
    cells = gpd.GeoDataFrame(
        {"cell": np.arange(len(lat) * len(lon))},
        geometry=[box(x - dlon / 2, y - dlat / 2, x + dlon / 2, y + dlat / 2) for y in lat for x in lon], crs=4326)
    overlap = gpd.overlay(cells.to_crs(6933), polygons[["province", "geometry"]].to_crs(6933), how="intersection")
    overlap["area"] = overlap.area
    provinces = sorted(polygons.province)
    weights = np.zeros((len(provinces), len(cells)))
    for (province, cell), area in overlap.groupby(["province", "cell"]).area.sum().items():
        weights[provinces.index(province), cell] = area
    weights /= weights.sum(axis=1, keepdims=True)
    return provinces, weights


def seconds_in_month(date):
    return calendar.monthrange(date.year, date.month)[1] * 86400


def main():
    import xarray as xr
    log = open_log(DATA_DIR / "02_province_members.log")
    files = sorted(GRIB_DIR.glob("seas5_tprate_lead1-6_*.grib"))
    assert len(files) == len(INIT_YEARS), f"expected {len(INIT_YEARS)} GRIB files, found {len(files)}"
    provinces, weights, grid = None, None, None
    parts = []
    for path in files:
        ds = xr.open_dataset(path, **GRIB_OPTIONS)
        if weights is None:
            provinces, weights = province_area_weights(ds.latitude.values, ds.longitude.values)
            grid = (ds.latitude.values.copy(), ds.longitude.values.copy())
        assert np.array_equal(ds.latitude.values, grid[0]) and np.array_equal(ds.longitude.values, grid[1]), f"grid differs: {path.name}"
        rate = ds["tprate"].transpose("number", "forecastMonth", "time", "latitude", "longitude").values     # m/s
        n_mem, n_lead, n_init = rate.shape[:3]
        inits = pd.to_datetime(ds.time.values)
        leads = ds.forecastMonth.values.astype(int)
        target = np.array([[i + pd.DateOffset(months=int(lead) - 1) for i in inits] for lead in leads])  # (lead, init)
        seconds = np.vectorize(seconds_in_month)(target)
        rain = rate * 1000.0 * seconds[None, :, :, None, None]                                             # mm per month
        rain_province = rain.reshape(n_mem, n_lead, n_init, -1) @ weights.T                                # (member, lead, init, province)
        index = pd.MultiIndex.from_product([range(n_mem), range(n_lead), range(n_init), range(len(provinces))],
                                           names=["member", "lead_idx", "init_idx", "province_idx"])
        df = pd.DataFrame({"rain_mm": rain_province.reshape(-1).astype("float32")}, index=index).reset_index()
        df["member"] = df.member.astype("int16")
        df["lead"] = leads[df.lead_idx.values].astype("int8")
        df["init"] = inits[df.init_idx.values]
        df["date"] = pd.to_datetime(target[df.lead_idx.values, df.init_idx.values])
        df["province"] = np.array(provinces)[df.province_idx.values]
        parts.append(df[["province", "init", "lead", "date", "member", "rain_mm"]])
        ds.close()
    members = pd.concat(parts, ignore_index=True)
    members["province"] = members.province.astype("category")
    members.to_parquet(SEAS5_MEMBERS, index=False)

    ensemble = (members.groupby(["province", "init", "lead", "date"], observed=True).rain_mm
                .agg(n_mem="size", ens_mean="mean", ens_sd=lambda s: float(np.std(s, ddof=1))).reset_index())
    ensemble.to_parquet(SEAS5_ENSEMBLE, index=False)

    n_expected = 38 * 12 * len(INIT_YEARS) * 6
    log(f"member rows: {len(members)} | ensemble rows: {len(ensemble)} (expected {n_expected})")
    log("initialisations:", members.init.min().date(), "to", members.init.max().date(),
        "| target months:", members.date.min().date(), "to", members.date.max().date())
    log("members per initialisation, up to 2016:", sorted(int(n) for n in ensemble[ensemble.init.dt.year <= 2016].n_mem.unique()),
        "| from 2017:", sorted(int(n) for n in ensemble[ensemble.init.dt.year >= 2017].n_mem.unique()))
    log("negative values:", int((members.rain_mm < 0).sum()), "| missing values:", int(members.rain_mm.isna().sum()))

    # mean bias of the raw ensemble mean against CHIRPS, all provinces and months 1986-2024
    obs = pd.read_csv(OBS_PANEL)
    obs["date"] = pd.to_datetime(obs.year_month + "-01")
    joined = ensemble.assign(province=lambda d: d.province.astype(str)).merge(obs[["province", "date", "rainfall"]], on=["province", "date"])
    joined = joined[joined.date.dt.year.between(1986, 2024)]
    for lead, g in joined.groupby("lead"):
        corr = np.corrcoef(g.ens_mean, g.rainfall)[0, 1]
        log(f"  lead {lead}: SEAS5 mean {g.ens_mean.mean():6.1f} mm, CHIRPS mean {g.rainfall.mean():6.1f} mm, "
            f"bias {g.ens_mean.mean() - g.rainfall.mean():+6.1f} mm, correlation incl. seasonal cycle {corr:.3f}, n = {len(g)}")
    log.close()


if __name__ == "__main__":
    main()
    # eccodes can abort while the interpreter shuts down, after all outputs are written; exit directly instead.
    sys.stdout.flush()
    os._exit(0)
