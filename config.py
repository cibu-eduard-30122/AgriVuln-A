"""
Configuration file for the AgriVuln‑AI project.

This module centralises settings such as the Google Earth Engine (GEE)
project, folder structure and helper functions to initialise GEE.  When
running any of the scripts in this repository, import the variables
defined here rather than hard‑coding paths in multiple places.  This
makes it easy to adjust the output location or change the GEE project
without touching every script.
"""

from pathlib import Path

try:
    import ee  # type: ignore
except ImportError:
    ee = None  # GEE will be initialised in scripts
    # baza proiectului
BASE_DIR = Path(__file__).resolve().parent

# directoare de date
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

# 👉 adaugă asta pentru figuri / hărți
FIGURES_DIR = BASE_DIR / "figures"

###############################################################################
# Google Earth Engine project settings
#
# Replace ``YOUR_EE_PROJECT`` with the Cloud project ID you used when
# registering for Earth Engine.  The project is required for paid use or
# large‑scale exports.  For free use, a project can still help organise
# tasks.
###############################################################################

PROJECT: str = "agrivuln2025"

###############################################################################
# Directory structure
#
# Raw data (GeoTIFFs) will be written to ``data/raw``, pre‑processed
# NetCDF files to ``data/processed`` and model results (trained models,
# metrics, SHAP values, figures) to ``results``.  The ``.mkdir`` calls
# ensure the folders exist when the scripts run.
###############################################################################

# Root of the repository (assumed to be the parent of this file)
ROOT_DIR: Path = Path(__file__).resolve().parent
# Data and results directories
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = ROOT_DIR / "results"

# Create directories if they don't exist
for p in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

###############################################################################
# Helper functions
###############################################################################

def initialize_ee() -> None:
    """Authenticate and initialise the Earth Engine API.

    The Earth Engine Python API requires authentication.  When run from
    an interactive environment like Colab or a notebook, this function
    will prompt you to follow a URL and paste the authentication code.
    Afterwards, the credentials are cached in ``~/.config/earthengine``.
    You should call this function at the start of your scripts that use
    ee.* functions.  The ``PROJECT`` variable will be passed to
    ``ee.Initialize`` to associate tasks with your Cloud project.
    """
    if ee is None:
        raise ImportError(
            "earthengine-api is not installed. Install it with `pip install earthengine-api`"
        )
    # Authenticate the user (this opens a browser window on first run)
    ee.Authenticate()
    # Initialise the API with the specified project
    ee.Initialize(project=PROJECT)