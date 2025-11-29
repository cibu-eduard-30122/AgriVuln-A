# add_extra_indices.py
#
# Scop:
#  - folosim ground_truth.csv (lat, lon, label)
#  - folosim features_ground_truth.csv (NDVI/EVI/CHIRPS/PM25 etc.)
#  - ne conectăm la Earth Engine
#  - extragem INDICI EXTRA la aceleași puncte:
#       * WorldPop_2020
#       * Water_occurrence (JRC)
#       * CAMS_PM25_JanMar2023
#       * Elevation_SRTM
#       * Slope_deg
#       * LandCover_2020 (ESA WorldCover)
#       * Dist_to_water_km
#  - salvăm un nou CSV: features_ground_truth_with_extra.csv

import os
import pandas as pd
import ee

from config import RAW_DIR, PROCESSED_DIR, initialize_ee


# -------------------------------
# 1. Construim imaginea cu indici extra
# -------------------------------
def build_extra_indices_image(roi):
    """
    Creează un ee.Image cu mai multe benzi socio-eco / fizice, decupat pe ROI:
      - WorldPop_2020
      - Water_occurrence
      - CAMS_PM25_JanMar2023
      - Elevation_SRTM
      - Slope_deg
      - LandCover_2020
      - Dist_to_water_km
    """
    # --- WorldPop 2020 (densitate populație, ~100m) ---
    worldpop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterDate("2020-01-01", "2021-01-01")
        .mean()
        .rename("WorldPop_2020")
    )

    # --- Water occurrence (JRC Global Surface Water) ---
    water_occ = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .rename("Water_occurrence")
    )

    # --- CAMS PM2.5 medie Jan–Mar 2023 ---
    cams_ic = (
        ee.ImageCollection("ECMWF/CAMS/NRT")
        .filterDate("2023-01-01", "2023-04-01")
    )
    # bandul corect pentru PM2.5 în dataset este "particulate_matter_d_less_than_25_um_surface"
    cams_pm25 = (
        cams_ic.select("particulate_matter_d_less_than_25_um_surface")
        .mean()
        .rename("CAMS_PM25_JanMar2023")
    )

    # --- SRTM Elevation + slope ---
    # ID corect: USGS/SRTMGL1_003
    srtm = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation_SRTM")
    slope = ee.Terrain.slope(srtm).rename("Slope_deg")

    # --- ESA WorldCover 2020: land cover ---
    worldcover_ic = ee.ImageCollection("ESA/WorldCover/v100")
    worldcover_img = (
        worldcover_ic.first()
        .select("Map")
        .rename("LandCover_2020")
    )

    # --- Distance to water (km) ---
    water_mask = water_occ.gte(50)          # >=50% occurrence => apă
    dist_px = water_mask.Not().fastDistanceTransform(30).sqrt()
    dist_km = dist_px.multiply(30).divide(1000).rename("Dist_to_water_km")

    # combinăm toate benzile într-o singură imagine și decupăm la ROI
    extra_img = ee.Image.cat(
        [
            worldpop,
            water_occ,
            cams_pm25,
            srtm,
            slope,
            worldcover_img,
            dist_km,
        ]
    ).clip(roi)

    return extra_img


# -------------------------------
# 2. Sampling la punctele de ground truth
# -------------------------------
def sample_new_indices_with_sampleRegions(
    img: ee.Image,
    gt: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    scale: int = 1000,
) -> pd.DataFrame:
    """
    Folosește ee.Image.sampleRegions (server-side) pentru a extrage valorile
    tuturor benzilor din `img` la punctele (lat, lon) din ground truth.

    Returnează un DataFrame cu o coloană "point_id" + benzile noi.
    """
    if lat_col not in gt.columns or lon_col not in gt.columns:
        raise ValueError(
            f"Ground truth trebuie să aibă coloanele '{lat_col}' și '{lon_col}'. "
            f"Coloane găsite: {list(gt.columns)}"
        )

    # construim FeatureCollection de puncte
    features = []
    for idx, row in gt.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        geom = ee.Geometry.Point([lon, lat])
        feat = ee.Feature(geom, {"point_id": int(idx)})
        features.append(feat)

    fc_points = ee.FeatureCollection(features)

    # sampleRegions pe server
    print("📌 Sampling new indices with sampleRegions (server-side)...")
    sampled_fc = img.sampleRegions(
        collection=fc_points,
        scale=scale,
        geometries=False,
    )

    sampled_dict = sampled_fc.getInfo()

    records = []
    for f in sampled_dict["features"]:
        props = f["properties"]
        point_id = int(props.pop("point_id"))
        rec = {"point_id": point_id}
        rec.update(props)
        records.append(rec)

    df_new = pd.DataFrame.from_records(records)
    df_new = df_new.sort_values("point_id").reset_index(drop=True)

    print(f"✅ New indices DataFrame shape: {df_new.shape}")
    print(f"✅ New index columns: {list(df_new.columns)}")
    return df_new


# -------------------------------
# 3. Main
# -------------------------------
def main():
    print(os.getcwd())

    # 3.1 Init EE
    print("🔑 Initializing Earth Engine...")
    initialize_ee()

    # 3.2 Ground truth + features existente
    gt_path = RAW_DIR / "ground_truth.csv"
    base_feat_path = PROCESSED_DIR / "features_ground_truth.csv"

    print(f"📂 Loading ground truth from {gt_path}")
    gt = pd.read_csv(gt_path)

    print(f"📂 Loading existing features from {base_feat_path}")
    base_df = pd.read_csv(base_feat_path)

    if len(gt) != len(base_df):
        raise ValueError(
            f"Numărul de rânduri din ground_truth ({len(gt)}) "
            f"nu coincide cu features_ground_truth ({len(base_df)})."
        )

    # 3.3 ROI = bounding box peste toate punctele
    roi = ee.Geometry.Rectangle(
        [
            float(gt["lon"].min()) - 0.1,
            float(gt["lat"].min()) - 0.1,
            float(gt["lon"].max()) + 0.1,
            float(gt["lat"].max()) + 0.1,
        ]
    )

    # 3.4 Construim imaginea cu indici extra
    print("🛰️ Building Earth Engine image for extra indices...")
    extra_img = build_extra_indices_image(roi)

    # 3.5 Sampling la puncte
    new_indices_df = sample_new_indices_with_sampleRegions(
        extra_img,
        gt,
        lat_col="lat",
        lon_col="lon",
        scale=1000,  # ~1 km rezoluție sampling
    )

    # 3.6 Combinăm cu features existente folosind point_id (JOIN pe stânga)
    #     - păstrăm TOATE cele 4925 de rânduri
    #     - punem NaN unde nu există date noi (puncte fără acoperire)
    base_df = base_df.reset_index().rename(columns={"index": "point_id"})
    # new_indices_df are deja point_id + coloanele noi

    cols_to_add = [c for c in new_indices_df.columns if c != "point_id"]

    merged = base_df.merge(new_indices_df, on="point_id", how="left")
    merged = merged.drop(columns=["point_id"])

    out_path = PROCESSED_DIR / "features_ground_truth_with_extra.csv"
    merged.to_csv(out_path, index=False)
    print(f"💾 Saved merged features with extra indices to: {out_path}")


if __name__ == "__main__":
    main()
