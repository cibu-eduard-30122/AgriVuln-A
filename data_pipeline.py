"""
Download and export multi‑indicator raster stacks from Google Earth Engine.

This script uses the Earth Engine Python API to collect monthly composites of
vegetation, soil moisture, precipitation, air quality and population
indicators over a given region.  It writes each monthly stack as a GeoTIFF
file in the directory specified by ``config.RAW_DIR``.

The indicators included in this example are:

  * **NDVI** and **EVI** from Sentinel‑2 MSI Level‑2A (`COPERNICUS/S2_SR`).
  * **Soil moisture** from SMAP (`NASA_USDA/HSL/SMAP10KM_soil_moisture`).
  * **Precipitation** from CHIRPS (`UCSB-CHG/CHIRPS/DAILY`).
  * **PM2.5** from CAMS (`ECMWF/CAMS/NRT`).
  * **Population density** from WorldPop (`WorldPop/GP/100m/population`).

You can add or remove indicators by editing the functions below.  Export
tasks are submitted to Earth Engine and will run asynchronously in the
background.  You can monitor progress in the Earth Engine Tasks page:
https://code.earthengine.google.com/tasks

To use this script in a Colab notebook, first mount your Google Drive and
install the dependencies listed in ``requirements.txt``.  Then run:

    %cd /content/drive/MyDrive/path/to/AgriVulnAI
    !python data_pipeline.py

Make sure you have authenticated with ``ee.Authenticate()`` and called
``ee.Initialize(project=config.PROJECT)`` before running this script.
"""

import datetime
from typing import List
import ee  # type: ignore
import geemap  # type: ignore

import config


def get_monthly_date_ranges(start_date: str, end_date: str) -> List[tuple]:
    """Return a list of (start, end) date tuples for each month in the range.

    Args:
        start_date: ISO‑format string (e.g. ``"2023-01-01"``).
        end_date: ISO‑format string (e.g. ``"2023-12-31"``).

    Returns:
        List of (start, end) date strings delimiting each calendar month.
    """
    start = datetime.datetime.fromisoformat(start_date)
    end = datetime.datetime.fromisoformat(end_date)
    ranges: List[tuple] = []
    current = datetime.date(start.year, start.month, 1)
    while current <= end.date():
        next_month = (current.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        last_day = next_month - datetime.timedelta(days=1)
        ranges.append((current.isoformat(), last_day.isoformat()))
        current = next_month
    return ranges


def compute_veg_indices(region: ee.Geometry, start: str, end: str) -> ee.Image:
    """Compute NDVI and EVI composites from Sentinel‑2 Surface Reflectance over the given period.

    This function uses the harmonized Sentinel‑2 Level‑2A collection, which removes
    processing baseline offsets present in the older ``COPERNICUS/S2_SR`` collection.
    The harmonized collection is recommended by the Earth Engine team【383876585094331†L72-L83】.

    Returns an image with two bands: ``NDVI`` and ``EVI``.
    """
    # Use the harmonized Sentinel‑2 SR collection to avoid deprecation warnings and
    # ensure consistent scaling across processing baselines【383876585094331†L72-L83】.
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    def add_indices(img):
        # Compute NDVI and EVI.  Sentinel‑2 bands: B4=red, B8=nir, B2=blue.
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        evi = img.expression(
            "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
            {
                "NIR": img.select("B8"),
                "RED": img.select("B4"),
                "BLUE": img.select("B2"),
            },
        ).rename("EVI")
        return img.addBands([ndvi, evi])

    with_indices = s2.map(add_indices)
    composite = (
        with_indices
        .select(["NDVI", "EVI"])
        .median()
    )
    return composite


def compute_soil_moisture(region: ee.Geometry, start: str, end: str) -> ee.Image:
    """Compute average soil moisture from SMAP over the period."""
    smap = (
        ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")
        .filterDate(start, end)
        .filterBounds(region)
    )
    # Band 'ssm' stands for surface soil moisture
    return smap.select("ssm").mean().rename("SMAP_soil_moisture")


def compute_precip(region: ee.Geometry, start: str, end: str) -> ee.Image:
    """Compute total precipitation from CHIRPS during the period."""
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(region)
    )
    # CHIRPS daily precipitation is in mm; sum to get monthly total
    return chirps.select("precipitation").sum().rename("CHIRPS_precip")


def compute_pm25(region: ee.Geometry, start: str, end: str) -> ee.Image:
    """Compute mean surface PM2.5 concentration from CAMS.

    The CAMS near‑real‑time dataset uses a nonstandard band name for fine
    particulate matter.  According to the Earth Engine catalog, the band
    ``particulate_matter_d_less_than_25_um_surface`` represents PM2.5
    concentrations【289722020921300†L114-L130】.  The earlier name ``pm2p5``
    is no longer available, which caused export tasks to fail when selecting
    ``pm2p5``.  Use the long band name instead and rename it to ``PM25`` for
    consistency.
    """
    cams = (
        ee.ImageCollection("ECMWF/CAMS/NRT")
        .filterDate(start, end)
        .filterBounds(region)
    )
    # Select the PM2.5 band.  The dataset uses ``25`` instead of ``2p5`` in the name.
    return cams.select("particulate_matter_d_less_than_25_um_surface").mean().rename("PM25")


def compute_population(region: ee.Geometry) -> ee.Image:
    """Return population density from the WorldPop global collection for a given region.

    The original script referenced the asset ``WorldPop/GP/100m/population/2020``, which
    is no longer available in the Earth Engine catalog.  The WorldPop project
    distributes an ImageCollection called ``WorldPop/GP/100m/pop`` that contains
    estimated total population counts for multiple years at 100 m resolution【440162229041465†L74-L88】.  Each
    image in the collection has a ``year`` property identifying the census year.  Here we
    select the 2020 layer, clip it to the region of interest, and rename the band.

    Args:
        region: The region of interest (ee.Geometry).

    Returns:
        An ``ee.Image`` with a single band ``Population`` representing population counts.
    """
    # Load the WorldPop population ImageCollection.
    collection = ee.ImageCollection("WorldPop/GP/100m/pop")
    # Filter the collection to the year 2020.  Other years (e.g., 2010, 2015) are available.
    pop_2020 = collection.filter(ee.Filter.eq("year", 2020)).first()
    # Clip to the region of interest and rename the band to a consistent name.
    return pop_2020.clip(region).rename("Population")


def export_monthly_stack(region: ee.Geometry, start: str, end: str, out_path: str) -> None:
    """Compute the multi‑indicator stack and export it as GeoTIFF.

    Args:
        region: The region of interest (ee.Geometry).
        start: Start date (inclusive).
        end: End date (inclusive).
        out_path: Local path (in Google Drive) where the GeoTIFF will be saved.
    """
    veg = compute_veg_indices(region, start, end)
    soil = compute_soil_moisture(region, start, end)
    precip = compute_precip(region, start, end)
    pm = compute_pm25(region, start, end)
    pop = compute_population(region)
    # Combine bands into a single stack.  Cast all bands to a common 32‑bit float
    # to ensure type consistency across the exported image.  Earth Engine will
    # otherwise raise an error if bands have mixed dtypes (e.g. Float32 vs
    # Float64)【289722020921300†L114-L130】.
    stack = veg.addBands([soil, precip, pm, pop]).toFloat()
    # Define CRS/resolution.  Use 100 m (EPSG:4326 approximate ~ 0.001 deg) for demonstration.
    reprojected = stack.reproject(crs="EPSG:4326", scale=100)
    # Export task
    task = ee.batch.Export.image.toDrive(
        image=reprojected,
        description=f"AgriVuln_stack_{start}",
        folder=out_path,
        fileNamePrefix=f"stack_{start}",
        region=region,
        scale=100,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Export started for {start} -> {end}, task id: {task.id}")


def main() -> None:
    # Initialise Earth Engine
    config.initialize_ee()
    # Define your region of interest here.  Example: bounding box around Kenya.
    # Kenya lies roughly between longitudes 33.97559E and 41.85688E and latitudes -4.47166S and 3.93726N【697500274376401†L17-L18】.
    # Adjust these coordinates if you wish to focus on a smaller area.
    roi = ee.Geometry.Rectangle([
        33.97559,  # West longitude (approximate western border of Kenya)
        -4.47166,  # South latitude (approximate southern border of Kenya)
        41.85688,  # East longitude (approximate eastern border of Kenya)
        3.93726   # North latitude (approximate northern border of Kenya)
    ])

    # Define time range
    START_DATE = "2023-01-01"
    END_DATE = "2023-03-31"
    months = get_monthly_date_ranges(START_DATE, END_DATE)

    # Loop over months and export each stack
    for (start, end) in months:
        # Output folder inside your Google Drive (under config.RAW_DIR).  In Colab you should
        # specify the Drive folder name (the Earth Engine export runs in your account).  Here
        # we use the name of the raw directory; Earth Engine will create the folder.
        export_monthly_stack(roi, start, end, out_path=config.RAW_DIR.name)


if __name__ == "__main__":
    main()