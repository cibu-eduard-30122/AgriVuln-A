# map_predictions_points.py
#
# Creează o hartă de vulnerabilitate pe punctele de ground_truth:
# - încarcă features_ground_truth_with_extra.csv
# - încarcă modelul LightGBM antrenat
# - face predicții pentru fiecare punct
# - salvează:
#     - CSV cu predicții
#     - GeoJSON pentru QGIS
#     - o hartă PNG (scatter) pentru raport

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path

from config import PROCESSED_DIR, RESULTS_DIR  # le avem deja în proiect


def main():
    base_dir = Path(".").resolve()
    print(base_dir)

    feat_path = PROCESSED_DIR / "features_ground_truth_with_extra.csv"
    model_path = RESULTS_DIR / "lightgbm_model.joblib"

    print(f"📂 Loading extended features from {feat_path}")
    df = pd.read_csv(feat_path)
    print(f"✅ DataFrame shape: {df.shape}")

    # --------- alegem coloanele de feature ----------
    # folosim tot ce NU este 'label'
    exclude_cols = ["label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"🔧 Using {len(feature_cols)} feature columns:")
    print(feature_cols)

    # --------- încărcăm modelul ----------
    print(f"📂 Loading trained model from {model_path}")
    model = joblib.load(model_path)

    # --------- predicții ----------
    print("🔮 Predicting vulnerability class for each point...")
    X = df[feature_cols]
    y_pred = model.predict(X)
    df["pred_class"] = y_pred

    # dacă modelul suportă predict_proba, salvăm și „încrederea”
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        df["pred_confidence"] = proba.max(axis=1)

    # --------- salvăm CSV ----------
    out_csv = PROCESSED_DIR / "ground_truth_with_predictions.csv"
    df.to_csv(out_csv, index=False)
    print(f"💾 Saved predictions CSV to: {out_csv}")

    # --------- GeoJSON + plot, dacă avem lat/lon ----------
    if "lon" in df.columns and "lat" in df.columns:
        print("🌍 Building GeoDataFrame...")
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )

        out_geojson = PROCESSED_DIR / "ground_truth_with_predictions.geojson"
        gdf.to_file(out_geojson, driver="GeoJSON")
        print(f"💾 Saved GeoJSON to: {out_geojson}")

        # --------- hartă scatter pentru raport ----------
        fig, ax = plt.subplots(figsize=(6, 6))
        gdf.plot(
            ax=ax,
            column="pred_class",
            cmap="viridis",
            markersize=5,
            legend=True,
            legend_kwds={"label": "Predicted vulnerability class"},
        )
        ax.set_title("AgriVuln-AI – predicted vulnerability (ground-truth points)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()

        out_png = RESULTS_DIR / "map_points_predictions.png"
        plt.savefig(out_png, dpi=300)
        plt.show()
        print(f"🖼️ Saved preview map PNG to: {out_png}")
    else:
        print("⚠️ Columns 'lat' and 'lon' not found; skipping GeoJSON and map plot.")


if __name__ == "__main__":
    main()
