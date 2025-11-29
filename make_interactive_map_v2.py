"""
make_interactive_map_v2.py

Construiește o hartă HTML interactivă pentru Kenya, folosind:
- ground_truth_with_predictions_v2.csv
- Vulnerability Index v2 (vuln_index_v2) pentru culoare/mărime
- Heatmap + puncte
- popup cu feature-uri cheie (fără 'NA', valorile lipsă devin 0)

Output:
  figures/kenya_vulnerability_index_v2.html
"""

from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.colormap import LinearColormap

# --------------------------------------------------
# 1. Path-uri (folosim config.py dacă există)
# --------------------------------------------------
try:
    from config import PROCESSED_DIR, FIGURES_DIR
    PROCESSED_DIR = Path(PROCESSED_DIR)
    FIGURES_DIR = Path(FIGURES_DIR)
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    FIGURES_DIR = BASE_DIR / "figures"

IN_PATH = PROCESSED_DIR / "ground_truth_with_predictions_v2.csv"
OUT_PATH = FIGURES_DIR / "kenya_vulnerability_index_v2.html"

print(f"📂 Input CSV:  {IN_PATH}")
print(f"📂 Output HTML: {OUT_PATH}")

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
if not IN_PATH.exists():
    raise FileNotFoundError(f"Nu găsesc fișierul: {IN_PATH}")

df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df):,} rows.")
print("Columns:", list(df.columns))

# verificăm coloane de bază
for col in ["lat", "lon"]:
    if col not in df.columns:
        raise ValueError(f"Lipsește coloana '{col}' din CSV.")

# Vulnerability score: folosim direct vuln_index_v2 dacă există
if "vuln_index_v2" in df.columns:
    df["vuln_score"] = df["vuln_index_v2"].astype(float)
else:
    # fallback: normalizare din pred_class
    if "pred_class" not in df.columns:
        raise ValueError("Nu există nici 'vuln_index_v2', nici 'pred_class' în CSV.")
    cmin, cmax = df["pred_class"].min(), df["pred_class"].max()
    if cmax == cmin:
        df["vuln_score"] = 0.5
    else:
        df["vuln_score"] = (df["pred_class"] - cmin) / (cmax - cmin)

# clamp în [0, 1]
df["vuln_score"] = df["vuln_score"].clip(0.0, 1.0)

print("\nVulnerability score stats (vuln_score):")
print(df["vuln_score"].describe())

# --------------------------------------------------
# 3. Pregătim colormap + centru hartă
# --------------------------------------------------
center_lat = float(df["lat"].mean())
center_lon = float(df["lon"].mean())
print(f"\n🌍 Map center: lat={center_lat:.3f}, lon={center_lon:.3f}")

# colormap: verde -> galben -> roșu pe [0,1]
colormap = LinearColormap(
    colors=["#2ecc71", "#f1c40f", "#e74c3c"],
    vmin=0.0,
    vmax=1.0,
)
colormap.caption = "Vulnerability Index v2 (0 = low, 1 = high)"

print(
    f"🎨 Colormap range vuln_score: {df['vuln_score'].min():.3f} – "
    f"{df['vuln_score'].max():.3f}"
)

# --------------------------------------------------
# 4. Construim harta Folium
# --------------------------------------------------
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="CartoDB dark_matter",
)

# Heatmap layer (pondere = vuln_score)
heat_data = df[["lat", "lon", "vuln_score"]].values.tolist()
HeatMap(
    heat_data,
    name="Vulnerability heatmap (v2)",
    radius=15,
    blur=20,
    max_zoom=10,
    min_opacity=0.3,
).add_to(m)


# --------------------------------------------------
# 4.1. Helper pt popup – fără 'NA'
# --------------------------------------------------
def build_popup_html(row) -> str:
    """
    Construiește HTML pentru popup.
    Valorile lipsă sunt înlocuite cu 0 (formatate frumos).
    """

    def get(col: str, ndigits: int = 3, default: float | int | None = 0.0):
        """Returnează valoarea formatată; dacă lipsă -> default (fără 'NA')."""
        if col not in row or pd.isna(row[col]):
            val = default
        else:
            val = row[col]

        if isinstance(val, (float, int, np.floating, np.integer)):
            if ndigits == 0:
                return f"{int(val)}"
            return f"{float(val):.{ndigits}f}"
        return str(val)

    parts = []

    cls = get("pred_class", ndigits=0, default=0)
    vuln = get("vuln_score", ndigits=2, default=0.0)
    idxv2 = get("vuln_index_v2", ndigits=3, default=0.0)

    parts.append(f"<b>Class (model):</b> {cls}<br/>")
    parts.append(f"<b>Vulnerability Index v2:</b> {idxv2} (vuln_score={vuln})<br/>")

    parts.append(
        f"<b>Lat:</b> {get('lat',3,default=0.0)}, "
        f"<b>Lon:</b> {get('lon',3,default=0.0)}<br/>"
    )

    feat_parts = []

    mapping = [
        ("NDVI_2023-03-01", "NDVI_2023-03-01", "#2ecc71", 3),
        ("EVI_2023-03-01", "EVI_2023-03-01", "#27ae60", 3),
        ("CHIRPS_precip_2023-03-01", "CHIRPS_2023-03-01", "#3498db", 3),
        ("PM25_2023-03-01", "PM25_2023-03-01", "#e74c3c", 3),
        ("CAMS_PM25_JanMar2023", "CAMS_PM25", "#c0392b", 3),
        ("Water_occurrence", "Water_occurrence", "#f1c40f", 3),
        ("WorldPop_2020", "WorldPop_2020", "#9b59b6", 0),
        ("pred_confidence", "pred_confidence", "#ecf0f1", 3),
    ]

    for col, label, color, ndigits in mapping:
        if col in df.columns:
            val_str = get(col, ndigits=ndigits, default=0.0)
            feat_parts.append(
                f'<span style="color:{color}">{label}: {val_str}</span>'
            )

    if feat_parts:
        parts.append("<b>Features:</b> " + " | ".join(feat_parts))

    return "".join(parts)


# --------------------------------------------------
# 4.2. Puncte colorate după vuln_score
# --------------------------------------------------
for _, row in df.iterrows():
    v = float(row["vuln_score"])
    color = colormap(v)
    radius = 3 + v * 7  # între ~3 și 10

    popup_html = build_popup_html(row)
    popup = folium.Popup(popup_html, max_width=450)

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        weight=0.5,
        popup=popup,
    ).add_to(m)

# adăugăm colormap ca legendă
colormap.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# --------------------------------------------------
# 5. Salvăm harta
# --------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
m.save(str(OUT_PATH))

print("\n✅ Map saved to:", OUT_PATH)
print("Poți deschide fișierul HTML în browser (double-click sau drag & drop).")
