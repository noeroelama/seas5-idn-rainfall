"""
Download ECMWF SEAS5 monthly mean precipitation rate over Indonesia from the Copernicus Climate Data Store (CDS):
dataset "seasonal-monthly-single-levels", all ensemble members, initialisations January 1985 to December 2024,
leadtime_month 1-6, 1 degree grid, area [7N, 94E, 12S, 142E]. One GRIB file per initialisation year.

For SEAS5, leadtime_month 1 is the initialisation month itself, so the forecast for target month t at lead L
comes from the initialisation of month t - L + 1. The hindcasts (1981-2016) have 25 members; the real-time
forecasts (from 2017) have 51.

Requirements: a CDS account with the dataset licence accepted and the API key in ~/.cdsapirc;
    pip install "cdsapi>=0.7.7" xarray cfgrib eccodes

Usage (from the project folder):
    python code/01_download_seas5.py                # all years not yet downloaded, 3 requests in parallel
    python code/01_download_seas5.py 1985 1986      # selected years
    python code/01_download_seas5.py --workers 1    # one request at a time
    python code/01_download_seas5.py --check        # check existing files only

Each file is written to a .part file and renamed when complete, so the script can be stopped and restarted.
After downloading, every file is checked (leads 1-6, 12 initialisations, 25 or 51 members, no missing or negative
values). Logs: data/c3s_seas5_lead1to6/download_log.csv and check_report.txt.
"""
import argparse
import csv
import importlib.util
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from config import GRIB_DIR, INIT_YEARS, LEADS

AREA = [7, 94, -12, 142]                # north, west, south, east
SYSTEMS = ["51", "5"]                   # SEAS5 system numbers tried in this order
ATTEMPTS_PER_SYSTEM = 3
_log_lock = threading.Lock()


def grib_path(year):
    return GRIB_DIR / f"seas5_tprate_lead1-6_{year}.grib"


def append_download_log(**fields):
    path = GRIB_DIR / "download_log.csv"
    new_file = not path.exists()
    with _log_lock, open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["time", "year", "system", "status", "bytes", "seconds", "message"])
        if new_file:
            writer.writeheader()
        writer.writerow({k: fields.get(k, "") for k in writer.fieldnames})


def download_year(year):
    import cdsapi
    target = grib_path(year)
    if target.exists() and target.stat().st_size > 0:
        return year, "already present"
    part = target.with_suffix(".grib.part")
    request = {
        "originating_centre": "ecmwf",
        "variable": ["total_precipitation"],
        "product_type": ["monthly_mean"],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "leadtime_month": [str(lead) for lead in LEADS],
        "data_format": "grib",
        "area": AREA,
    }
    client = cdsapi.Client(quiet=True, progress=False)             # one client per thread
    for system in SYSTEMS:
        for attempt in range(1, ATTEMPTS_PER_SYSTEM + 1):
            t0 = time.time()
            now = datetime.now().isoformat(timespec="seconds")
            try:
                client.retrieve("seasonal-monthly-single-levels", dict(request, system=system), str(part))
                os.replace(part, target)
                size = target.stat().st_size
                append_download_log(time=now, year=year, system=system, status="ok", bytes=size, seconds=round(time.time() - t0))
                return year, f"ok (system {system}, {size / 1e6:.1f} MB, {time.time() - t0:.0f} s)"
            except Exception as exc:
                append_download_log(time=now, year=year, system=system, status=f"failed {attempt}",
                                    seconds=round(time.time() - t0), message=str(exc).replace("\n", " ")[:300])
                if part.exists():
                    part.unlink()
                time.sleep(30 * attempt)                            # longer wait after each failure
    return year, "FAILED, see download_log.csv"


def check_year(year):
    """Return a list of problems found in the file of one initialisation year (empty list: file is fine)."""
    import xarray as xr
    path = grib_path(year)
    if not path.exists():
        return ["file missing"]
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs=dict(time_dims=("forecastMonth", "time"), indexpath=""))
    rate = ds["tprate"]
    problems = []
    if list(map(int, ds.forecastMonth.values)) != list(LEADS):
        problems.append(f"leads {list(ds.forecastMonth.values)}")
    inits = ds.time.values
    if len(inits) != 12 or str(inits[0])[:7] != f"{year}-01" or str(inits[-1])[:7] != f"{year}-12":
        problems.append(f"{len(inits)} initialisations, {str(inits[0])[:7]} to {str(inits[-1])[:7]}")
    expected_members = 25 if year <= 2016 else 51
    if ds.sizes["number"] != expected_members:
        problems.append(f"{ds.sizes['number']} members (expected {expected_members})")
    n_missing = int(rate.isnull().sum())
    if n_missing:
        problems.append(f"{n_missing} missing values")
    if float(rate.min()) < 0:
        problems.append(f"negative values (min {float(rate.min()):.2e})")
    ds.close()
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("years", nargs="*", type=int)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--check", action="store_true", help="check existing files only")
    args = parser.parse_args()
    years = args.years or list(INIT_YEARS)
    GRIB_DIR.mkdir(parents=True, exist_ok=True)

    if not args.check:
        missing_years = [y for y in years if not (grib_path(y).exists() and grib_path(y).stat().st_size > 0)]
        print(f"{len(years) - len(missing_years)} years present, {len(missing_years)} to download, {args.workers} in parallel")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for future in as_completed([pool.submit(download_year, y) for y in missing_years]):
                year, status = future.result()
                print(f"  {year}: {status}", flush=True)

    not_installed = [m for m in ("xarray", "cfgrib") if importlib.util.find_spec(m) is None]
    if not_installed:
        print(f"\nCheck skipped: {', '.join(not_installed)} not installed. Install xarray, cfgrib and eccodes, "
              "then run with --check. check_report.txt was not changed.")
        sys.exit(2)
    print("checking files ...")
    lines, n_bad = [], 0
    for year in years:
        try:
            problems = check_year(year)
        except Exception as exc:
            problems = [f"cannot be read: {str(exc)[:150]}"]
        n_bad += bool(problems)
        lines.append(f"{year}: " + ("ok" if not problems else "; ".join(problems)))
        print("  " + lines[-1])
    (GRIB_DIR / "check_report.txt").write_text(
        f"Checked {datetime.now().isoformat(timespec='seconds')}: {len(years) - n_bad} ok, {n_bad} with problems\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"done: {len(years) - n_bad} ok, {n_bad} with problems")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
