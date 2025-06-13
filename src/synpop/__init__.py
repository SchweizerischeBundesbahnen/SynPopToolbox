"""SynPopToolbox - A set of tools to analyse, modify or extend the Swiss synthetic population."""
# New logger: Change level between INFO or DEBUG to have more or less information
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s"
)

# Mute some other loggers
logging.getLogger("fiona").setLevel(logging.CRITICAL)
logging.getLogger("shapely").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("pyproj").setLevel(logging.CRITICAL)
logging.getLogger("crs").setLevel(logging.CRITICAL)

# constants
DEFAULT_GEO_COLS = [
    "kt_name",
    "kt_id",
    "mun_name",
    "mun_id",
    "sl3_name",
    "agglo_name",
    "agglo_id",
    "amgr_name",
    "amgr_id",
    "amr_name",
    "amr_id",
    "msr_name",
    "msr_id",
    "sl3_id",
    "zone_id",
]

HOME_DIR = Path(__file__).parent

NEW_CATEGORIES = {
    "current_job_rank": [
        "apprentice",
        "employee",
        "management",
        "null",
    ],
    "current_edu": [
        "kindergarten",
        "apprentice",
        "pupil_primary",
        "pupil_secondary",
        "student",
        "null",
    ],
    "highest_education": [
        "primary",
        "secondary",
        "tertiary_college",
        "tertiary_university",
    ],
}


def mobi_zones_path() -> Path:
    """Get path to MOBi Zones shapes. Download from DVC if needed."""
    mobi_zones = HOME_DIR / "assets/resources/geodata/zones.gpkg"
    return mobi_zones
