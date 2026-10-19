# Gradient boosting correction of SEAS5 monthly rainfall forecasts for 38 provinces in Indonesia

Code and data for the manuscript "Gradient boosting correction of SEAS5 monthly rainfall forecasts for 38 provinces
in Indonesia: A comparison with quantile mapping" (Arif Nurwahid, Ehime University; submitted to *Array*).

The study compares LightGBM gradient boosting with six reference forecasts (climatology, damped anomaly persistence, raw
SEAS5, mean-bias correction, empirical quantile mapping and linear MOS) for monthly rainfall in the 38 provinces of
Indonesia, lead times of one to six months, in a rolling-origin test over 2005–2024.

## Folders

```
code/                 scripts, run in the order below from the repository root
data/                 inputs and the SEAS5 province data
results/              forecasts, verification scores and logs
figures/              figures of the manuscript and the graphical abstract (PDF and PNG)
manuscript/tables/    LaTeX tables written by code/06_tables.py
manuscript/supplementary_tables.xlsx   Supplementary Tables S1–S4
```

## Inputs (data/)

| File | Content |
|---|---|
| `panel_rainfall_long.csv` | CHIRPS v2.0 rainfall averaged over each province (mm/month), Niño3.4 and DMI, January 1985 to April 2025 (only January 1985 to December 2024 is used) |
| `provinces_simplified.geojson` | polygons of the 38 provinces (2023 boundaries) |
| `a_geo_centroids.csv` | province centroids (rows with `country == "IDN"`) |
| `seas5_province_members_long.parquet` | SEAS5 monthly precipitation for each province, ensemble member, start date and lead (mm/month) |
| `seas5_province_ensemble.parquet` | ensemble mean and standard deviation of the above |
| `c3s_seas5_lead1to6/` | download log of the SEAS5 GRIB files; the GRIB files themselves are not included (download with script 01) |

## Scripts

| Script | Output | Time* |
|---|---|---|
| `01_download_seas5.py` | SEAS5 GRIB files, leads 1–6, start dates 1985–2024 (needs a Copernicus CDS account) | download |
| `02_province_members.py` | `data/seas5_province_members_long.parquet`, `data/seas5_province_ensemble.parquet` | 2 min |
| `03_forecasts.py` | `results/forecasts_amount.parquet`, `results/forecasts_prob.parquet`, `results/n_trees.csv` | 30 min |
| `04_verify.py` | skill scores (overall and by test year), tests and reliability data in `results/*.csv` | 1 min |
| `05_figures.py` | `figures/Fig1`–`Fig9` (PDF and PNG) | < 1 min |
| `06_tables.py` | `manuscript/tables/*.tex` (Tables 2 and 3), `manuscript/supplementary_tables.xlsx` (Tables S1–S4), `results/numbers.txt` | 1 min |
| `07_graphical_abstract.py` | `figures/graphical_abstract` (PDF and PNG) | < 1 min |

\* two CPU cores. `config.py` holds the paths, the study period and the province names; `scores.py` holds the
tercile categories and the ranked probability score.

The SEAS5 province data are included, so scripts 03–07 can be run without downloading SEAS5:

```bash
pip install -r requirements.txt
python code/03_forecasts.py
python code/04_verify.py
python code/05_figures.py
python code/06_tables.py
python code/07_graphical_abstract.py
```

## Software

Python 3.13.16, LightGBM 4.7.0, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, xarray 2026.9.0, cfgrib 0.9.15.1, GeoPandas 1.2.0,
Matplotlib 3.11.2, openpyxl 3.1.5, pyarrow 26.0.0 (`requirements.txt`). With the same versions and two threads,
`03_forecasts.py` reproduces the forecasts exactly (fixed random seed).

## Data sources

| Data | Source | Accessed |
|---|---|---|
| SEAS5 monthly precipitation, system 51 | Copernicus Climate Change Service, Climate Data Store, doi:10.24381/cds.68dd14c3 | 14 October 2026 |
| CHIRPS v2.0 monthly, 0.05° | Climate Hazards Center, UC Santa Barbara (Funk et al., 2015, doi:10.1038/sdata.2015.66) | 14 May 2026 |
| Niño3.4 and DMI (HadISST1.1) | NOAA Physical Sciences Laboratory, https://psl.noaa.gov/data/timeseries/month/ | 11 December 2025 |
| Province boundaries (2023) | batas-administrasi-indonesia, https://github.com/Alf-Anas/batas-administrasi-indonesia | 13 May 2026 |

Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is
responsible for any use that may be made of the Copernicus information or data it contains.

## Licence

Code: MIT licence (`LICENSE`). Data and results produced in this study: CC BY 4.0 (`DATA_LICENSE.md`); the source data
keep the terms of their providers listed above.

## Citation

See `CITATION.cff`. Please also cite the article once it is published.
