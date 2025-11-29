# export_features_csv.py
#
# Script care ia stack.nc + ground_truth.csv și produce
# data/processed/features_ground_truth.csv cu:
#   latitude, longitude, (alte coloane din ground truth),
#   toate feature-urile satelitare, label

import pandas as pd
import xarray as xr

from config import PROCESSED_DIR, RAW_DIR


def sample_features(
    ds: xr.Dataset,
    gt: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    label_col: str = "label",
):
    """
    Extrage valorile tuturor variabilelor din ds (stack.nc)
    la coordonatele (lat, lon) din ground truth.

    În stack.nc coordonatele se numesc:
      - y = latitudine
      - x = longitudine

    Returnează:
      X    - DataFrame cu feature-uri (NDVI_2023-01-01 etc)
      y    - vector de label-uri
      meta - DataFrame cu meta-informații din ground truth
    """

    # --- verificări de bază ---
    if lat_col not in gt.columns or lon_col not in gt.columns:
        raise ValueError(
            f"Ground truth trebuie să aibă coloanele '{lat_col}' și '{lon_col}'. "
            f"Coloane găsite: {list(gt.columns)}"
        )

    if label_col not in gt.columns:
        raise ValueError(
            f"Ground truth trebuie să aibă coloana '{label_col}' ca țintă (label). "
            f"Coloane găsite: {list(gt.columns)}"
        )

    for coord_name in ("y", "x"):
        if coord_name not in ds.coords:
            raise ValueError(
                f"Datasetul stack.nc nu are coordonata '{coord_name}'. "
                f"Coordonatele disponibile: {list(ds.coords.keys())}"
            )

    # --- pregătim vectorii de coordonate pentru sampling ---
    lats = xr.DataArray(gt[lat_col].values, dims=("points",))
    lons = xr.DataArray(gt[lon_col].values, dims=("points",))

    # --- pregătim structura pentru feature-uri ---
    feature_dict = {}

    # dacă avem coordonata time, o folosim pentru a pune datele în numele coloanelor
    has_time = "time" in ds.coords

    if has_time:
        times = pd.to_datetime(ds["time"].values)
    else:
        times = [None]

    # --- parcurgem fiecare variabilă (NDVI, EVI, CHIRPS_precip, PM25, Population) ---
    for var_name, da in ds.data_vars.items():
        dims = da.dims

        # cazul cel mai frecvent: variabilă cu time, y, x
        if "time" in dims and "y" in dims and "x" in dims:
            # selectăm toate punctele o dată (primim dims: time, points)
            sampled = da.sel(y=lats, x=lons, method="nearest")  # (time, points)
            sampled = sampled.transpose("points", "time")       # (points, time)
            vals = sampled.values                               # numpy array

            for t_idx, t_val in enumerate(times):
                if t_val is None:
                    col_name = var_name
                else:
                    col_name = f"{var_name}_{pd.to_datetime(t_val).strftime('%Y-%m-%d')}"
                feature_dict[col_name] = vals[:, t_idx]

        # variabilă fără time (statică) – ex. un layer static
        elif "y" in dims and "x" in dims:
            sampled = da.sel(y=lats, x=lons, method="nearest")  # (points,)
            vals = sampled.values
            feature_dict[var_name] = vals
        else:
            print(f"⚠️ Sar variabila {var_name} cu dims {dims} – pattern necunoscut.")

    # DataFrame cu toate feature-urile
    X = pd.DataFrame(feature_dict)

    # scoatem coloanele complet NaN (ex. Population_* dacă nu există valori reale)
    X = X.dropna(axis=1, how="all")

    # vectorul de label-uri
    y = gt[label_col].values

    # meta: toate coloanele din ground truth, mai puțin labelul
    meta_cols = [c for c in gt.columns if c != label_col]
    meta = gt[meta_cols].copy()

    return X, y, meta


def main():
    stack_path = PROCESSED_DIR / "stack.nc"
    gt_path = RAW_DIR / "ground_truth.csv"

    print(f"📂 Loading stack from {stack_path}")
    ds = xr.open_dataset(stack_path)

    print(f"📂 Loading ground truth from {gt_path}")
    gt = pd.read_csv(gt_path)

    print("🧪 Sampling satellite features la punctele de ground truth...")
    X, y, meta = sample_features(
        ds, gt,
        lat_col="lat",
        lon_col="lon",
        label_col="label",
    )

    print(f"✅ X shape: {X.shape}")
    print(f"✅ Primele coloane: {list(X.columns[:10])}")

    # combinăm meta + feature-uri + label
    df = pd.concat(
        [
            meta.reset_index(drop=True),
            X.reset_index(drop=True),
        ],
        axis=1,
    )
    df["label"] = y

    out_path = PROCESSED_DIR / "features_ground_truth.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Features CSV salvat la: {out_path}")


if __name__ == "__main__":
    main()
