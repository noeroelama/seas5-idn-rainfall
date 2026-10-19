"""Paths, study period and province names shared by the scripts 01-07."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GRIB_DIR = DATA_DIR / "c3s_seas5_lead1to6"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "manuscript" / "tables"

# Input files
OBS_PANEL = DATA_DIR / "panel_rainfall_long.csv"            # CHIRPS province rainfall (mm/month), Nino3.4 and DMI, 1985-01 to 2025-04
PROVINCE_POLYGONS = DATA_DIR / "provinces_simplified.geojson"
PROVINCE_CENTROIDS = DATA_DIR / "a_geo_centroids.csv"
SEAS5_MEMBERS = DATA_DIR / "seas5_province_members_long.parquet"
SEAS5_ENSEMBLE = DATA_DIR / "seas5_province_ensemble.parquet"

INIT_YEARS = range(1985, 2025)          # SEAS5 initialisation years
FIRST_TRAIN_YEAR = 1986                 # first target year used for training
TEST_YEARS = range(2005, 2025)          # rolling-origin test years
LEADS = range(1, 7)                     # lead 1 = initialisation month

# Province names in the polygon file (Indonesian) -> names used in the panel and the paper (English)
PROVINCE_NAME_EN = {
    "Aceh": "Aceh", "Sumatera Utara": "North Sumatra", "Sumatera Barat": "West Sumatra", "Riau": "Riau", "Jambi": "Jambi",
    "Sumatera Selatan": "South Sumatra", "Bengkulu": "Bengkulu", "Lampung": "Lampung",
    "Kepulauan Bangka Belitung": "Bangka Belitung Islands", "Kepulauan Riau": "Riau Islands", "DKI Jakarta": "Jakarta",
    "Jawa Barat": "West Java", "Jawa Tengah": "Central Java", "Daerah Istimewa Yogyakarta": "Yogyakarta", "Jawa Timur": "East Java",
    "Banten": "Banten", "Bali": "Bali", "Nusa Tenggara Barat": "West Nusa Tenggara", "Nusa Tenggara Timur": "East Nusa Tenggara",
    "Kalimantan Barat": "West Kalimantan", "Kalimantan Tengah": "Central Kalimantan", "Kalimantan Selatan": "South Kalimantan",
    "Kalimantan Timur": "East Kalimantan", "Kalimantan Utara": "North Kalimantan", "Sulawesi Utara": "North Sulawesi",
    "Sulawesi Tengah": "Central Sulawesi", "Sulawesi Selatan": "South Sulawesi", "Sulawesi Tenggara": "Southeast Sulawesi",
    "Gorontalo": "Gorontalo", "Sulawesi Barat": "West Sulawesi", "Maluku": "Maluku", "Maluku Utara": "North Maluku",
    "Papua": "Papua", "Papua Barat": "West Papua", "Papua Selatan": "South Papua", "Papua Tengah": "Central Papua",
    "Papua Pegunungan": "Highland Papua", "Papua Barat Daya": "Southwest Papua",
}


def open_log(path):
    """Return a function that prints a line and appends it to the log file at `path`."""
    fh = open(path, "w", encoding="utf-8")

    def log(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    log.close = fh.close
    return log
