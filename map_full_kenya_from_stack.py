"""
map_full_kenya_from_stack.py

Construiește o hartă PNG (full grid) pentru Kenya, folosind stack.nc:

- încarcă stack-ul de feature-uri (NDVI, EVI, CHIRPS_precip, PM25, Population)
- (opțional) încarcă modelul LightGBM (lightgbm_model.joblib) și prezice pred_class
- calculează Vulnerability Index v2 (0–1) pentru fiecare celulă
- desenează o hartă continuă (imshow) și o salvează ca PNG

Output:
  figures/kenya_vulnerability_grid_v2.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PATH-uri (config.py dacă există, altfel fallback)
# --------------------------------------------------
try:
    from config import PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR
    PROCESSED_DIR = Path(PROCESSED_DIR)
    RESULTS_DIR = Path(RESULTS_DIR)
    FIGURES_DIR = Path(FIGURES_DIR)
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = BASE_DIR / "figures"

STACK_PATH = PROCESSED_DIR / "stack.nc"
MODEL_PATH = RESULTS_DIR / "lightgbm_model.joblib"   # modelul tău
OUT_PNG = FIGURES_DIR / "kenya_vulnerability_grid_v2.png"

print(str(PROCESSED_DIR.parent))
print(f"📂 Stack:  {STACK_PATH}")
print(f"📂 Model:  {MODEL_PATH} (opțional)")
print(f"📂 Output PNG: {OUT_PNG}")

# --------------------------------------------------
# 2. Încărcăm stack-ul și agregăm pe timp
# --------------------------------------------------
if not STACK_PATH.exists():
    raise FileNotFoundError(f"Nu găsesc stack.nc la: {STACK_PATH}")

ds = xr.open_dataset(STACK_PATH)
print("✅ Opened stack.nc")
print("Original stack dims:", ds.dims)

# Coarsen spațial (y,x) și mediem pe timp (time)
COARSEN_FACTOR = 10
if "y" in ds.dims and "x" in ds.dims:
    ds_c = ds.coarsen(y=COARSEN_FACTOR, x=COARSEN_FACTOR, boundary="trim").mean()
    print(f"After coarsen (factor={COARSEN_FACTOR}) dims:", ds_c.dims)
else:
    ds_c = ds

if "time" in ds_c.dims:
    ds_c = ds_c.mean(dim="time")
    print("After mean over time dims:", ds_c.dims)

# --------------------------------------------------
# 3. Convertim la DataFrame (y,x -> rânduri)
# --------------------------------------------------
df = ds_c.to_dataframe().reset_index()
print(f"DataFrame shape before filtering: {df.shape}")
print("DataFrame columns:", df.columns.tolist())

# Coloanele reale din stack.nc:
# ['y', 'x', 'NDVI', 'EVI', 'CHIRPS_precip', 'PM25', 'Population']
wanted_cols = [
    "y",
    "x",
    "NDVI",
    "EVI",
    "CHIRPS_precip",
    "PM25",
    "Population",
]

available_cols = [c for c in wanted_cols if c in df.columns]
df = df[available_cols].copy()

print(f"Columns kept for computation: {available_cols}")
print(f"DataFrame shape after selecting columns: {df.shape}")

ny = ds_c.sizes["y"]
nx = ds_c.sizes["x"]
print(f"Grid size: n_y={ny}, n_x={nx}, total cells ~ {ny*nx:,}")

# --------------------------------------------------
# 4. Pregătim feature-urile și pred_class cu modelul
# --------------------------------------------------
feature_cols_for_model = [
    c
    for c in ["NDVI", "EVI", "CHIRPS_precip", "PM25", "Population"]
    if c in df.columns
]

X = df[feature_cols_for_model].copy()
X = X.fillna(X.median(numeric_only=True))  # completăm lipsurile

df["pred_class"] = 0  # default

if MODEL_PATH.exists():
    import joblib

    print("✅ Found model, loading:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    try:
        # ATENȚIE: modelul a fost antrenat cu 21 features,
        #   aici avem 5. Nu forțăm predicția ca să nu stricăm indexul.
        df["pred_class"] = model.predict(X)
        print("✅ Predicții pred_class făcute pentru grid.")
    except Exception as e:
        print("⚠️ Nu am reușit să fac predicții cu modelul:", e)
        print("   Continui fără pred_class (toate 0).")
else:
    print("⚠️ Nu am găsit modelul LightGBM. pred_class va fi 0 pentru toate celulele.")

# --------------------------------------------------
# 5. Calculăm Vulnerability Index v2 (0–1) – normalizare manuală
# --------------------------------------------------
risk_components = []

# 1) stress vegetație (1 - NDVI/EVI)
if "NDVI" in df.columns:
    risk_components.append(1.0 - X["NDVI"].astype(float))
if "EVI" in df.columns:
    risk_components.append(1.0 - X["EVI"].astype(float))

# 2) precipitații
if "CHIRPS_precip" in df.columns:
    risk_components.append(X["CHIRPS_precip"].astype(float))

# 3) poluare
if "PM25" in df.columns:
    risk_components.append(X["PM25"].astype(float))

# 4) densitate populație
if "Population" in df.columns:
    risk_components.append(X["Population"].astype(float))

# 5) clasa prezisă (dacă a reușit)
risk_components.append(df["pred_class"].astype(float))

scaled_list = []

for rc in risk_components:
    arr = np.asarray(rc, dtype=float)
    mask = np.isfinite(arr)

    if not mask.any():
        scaled = np.zeros_like(arr)
    else:
        amin = arr[mask].min()
        amax = arr[mask].max()
        if amax > amin:
            scaled = (arr - amin) / (amax - amin)
        else:
            scaled = np.zeros_like(arr)

    scaled_list.append(scaled)

stack_scaled = np.vstack(scaled_list)
vuln_index_v2 = np.nanmean(stack_scaled, axis=0)
df["vuln_index_v2"] = vuln_index_v2

print("\n📈 Vulnerability Index v2 stats (0–1):")
print(df["vuln_index_v2"].describe())

# --------------------------------------------------
# 6. Reconstruim grila 2D corect și desenăm PNG
#    (fără reindex artificial la 0..ny-1, folosim coordonatele reale din stack)
# --------------------------------------------------
if not {"y", "x"}.issubset(df.columns):
    raise ValueError("Nu găsesc 'y' și 'x' în DataFrame pentru reconstrucția grilei.")

# pivotăm pe (y,x) -> matrice
pivot = (
    df.set_index(["y", "x"])["vuln_index_v2"]
      .sort_index()              # sortăm după y
      .unstack("x")              # coloane = x
      .sort_index(axis=1)        # sortăm și coloanele după x
)

grid = pivot.values
y_coords = pivot.index.values
x_coords = pivot.columns.values

extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
print(f"\n🗺️ Extent (from coords): lon [{extent[0]}, {extent[1]}], lat [{extent[2]}, {extent[3]}]")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 10))
im = plt.imshow(
    grid,
    origin="lower",
    extent=extent,
    cmap="RdYlGn_r",    # verde = low, roșu = high
    vmin=0.0,
    vmax=1.0,
)
plt.colorbar(im, label="Vulnerability Index v2 (0 = low, 1 = high)")
plt.title("AgriVulnAI – Kenya Agricultural Vulnerability Index v2 (full grid)")
plt.xlabel("X (grid)")
plt.ylabel("Y (grid)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"\n✅ PNG salvat la: {OUT_PNG}")
