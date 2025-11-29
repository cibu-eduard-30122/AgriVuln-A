"""
Pre-process downloaded GeoTIFF stacks into a single NetCDF dataset.

This script scans the directory specified by ``config.RAW_DIR`` for GeoTIFF
files exported by ``data_pipeline.py``, stacks them along a new ``time``
dimension and writes the result to ``config.PROCESSED_DIR`` as a NetCDF file.

Each GeoTIFF exported by the pipeline contains multiple bands
(NDVI, EVI, CHIRPS_precip, PM25, Population, etc.).  The
timestamp of the stack is inferred from the filename
(e.g. ``stack_2023-01-01.tif`` -> time = 2023-01-01).

IMPORTANT:
The previous version built x/y coordinates incorrectly, which made all
ground-truth points sample the same pixel. Here we use rasterio.transform.xy
to compute proper longitude/latitude coordinates for each row and column.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr  # type: ignore
import rasterio  # type: ignore
from rasterio.transform import xy  # type: ignore

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_geotiffs(directory: Path) -> List[Path]:
    """Return a sorted list of GeoTIFF files in the directory."""
    tifs = [f for f in directory.glob("*.tif") if f.is_file()]
    return sorted(tifs)


def extract_date_from_filename(fname: str) -> str:
    """Extract date from filename pattern 'stack_YYYY-MM-DD.tif'."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    if not match:
        raise ValueError(f"Cannot extract date from filename {fname}")
    return match.group(1)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def preprocess_stacks(files: List[Path]) -> xr.Dataset:
    """
    Open each GeoTIFF and stack into an xarray.Dataset with a ``time`` dimension.

    For each file:
      - read the array with shape (band, height, width)
      - build 1D coordinate arrays:
            x -> longitudes for each column
            y -> latitudes  for each row
      - convert to an xarray Dataset with one variable per band

    Finally, all monthly datasets are concatenated along the ``time`` dimension.
    """
    datasets: list[xr.Dataset] = []

    for f in files:
        date_str = extract_date_from_filename(f.name)
        time_val = np.datetime64(date_str)

        with rasterio.open(f) as src:
            # arr shape: (bands, height, width)
            arr = src.read()

            height, width = src.height, src.width
            rows = np.arange(height)
            cols = np.arange(width)

            # rasterio.transform.xy(transform, rows, cols, offset='center')
            #  -> works cu arrays; returnează listă de coordonate
            # Longitudes: row = 0 (prima linie), toate coloanele
            xs, _ = xy(src.transform, 0 * cols, cols)
            xs = np.array(xs)

            # Latitudes: toate rândurile, col = 0 (prima coloană)
            _, ys = xy(src.transform, rows, 0 * rows)
            ys = np.array(ys)

            # Numele benzilor (dacă lipsesc, folosim band_1, band_2, ...)
            band_names = [
                src.descriptions[i] if src.descriptions[i] else f"band_{i+1}"
                for i in range(src.count)
            ]

            da = xr.DataArray(
                arr,
                dims=("band", "y", "x"),
                coords={
                    "band": band_names,
                    "y": ys,  # latitude
                    "x": xs,  # longitude
                },
            )

            ds = da.to_dataset(dim="band")
            # adăugăm o dimensiune time de mărime 1
            ds = ds.expand_dims(time=[time_val])
            datasets.append(ds)

    # concatenăm toate lunile pe dimensiunea time
    combined = xr.concat(datasets, dim="time")
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    files = list_geotiffs(config.RAW_DIR)
    if not files:
        raise FileNotFoundError(f"No GeoTIFFs found in {config.RAW_DIR}")
    print(f"Found {len(files)} GeoTIFF files. Processing...")

    ds = preprocess_stacks(files)

    out_path = config.PROCESSED_DIR / "stack.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)

    print(f"Saved NetCDF to {out_path}")


if __name__ == "__main__":
    main()
