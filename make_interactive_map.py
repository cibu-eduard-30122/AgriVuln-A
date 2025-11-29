# make_interactive_map.py
#
# Harta interactivă "master" a punctelor de vulnerabilitate pentru Kenya
# - citește ground_truth_with_predictions.csv din PROCESSED_DIR
# - generează HTML cu:
#     * puncte colorate verde-galben-roșu după vuln_score
#     * heatmap
#     * legendă clară
#     * layer control simplu

import os
import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.colormap import LinearColormap

from config import PROCESSED_DIR, FIGURES_DIR


# === CONFIG ===
PREDICTIONS_CSV = os.path.join(
    PROCESSED_DIR, "ground_truth_with_predictions.csv"
)

OUTPUT_HTML = os.path.join(
    FIGURES_DIR, "kenya_vulnerability_points_master.html"
)


def load_predictions(path: str) -> pd.DataFrame:
    print(f"➡️  Loading predictions from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu găsesc fișierul: {path}")

    df = pd.read_csv(path)

    # verificăm coloanele esențiale
    required = ["lat", "lon", "pred_class"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Lipsește coloana obligatorie '{col}' din {path}. "
                f"Coloane disponibile: {list(df.columns)}"
            )

    print("First rows:")
    print(df[["lat", "lon", "pred_class"]].head())
    return df


def add_vulnerability_score(df: pd.DataFrame) -> pd.DataFrame:
    """creează o coloană vuln_score în [0, 1] pe baza pred_class."""
    print("\nℹ️  Construiesc vuln_score din pred_class (normalizare [0,1])")
    cmin = df["pred_class"].min()
    cmax = df["pred_class"].max()

    if cmax == cmin:
        df["vuln_score"] = 0.5
    else:
        df["vuln_score"] = (df["pred_class"] - cmin) / (cmax - cmin)

    print("vuln_score stats:")
    print(df["vuln_score"].describe())
    return df


def build_colormap(df: pd.DataFrame) -> LinearColormap:
    vmin = df["vuln_score"].min()
    vmax = df["vuln_score"].max()
    print(f"\n🎨 Colormap range vuln_score: {vmin:.3f} – {vmax:.3f}")

    # verde -> galben -> roșu
    colors = [
        "#1a9850",  # low
        "#fee08b",  # medium
        "#d73027",  # high
    ]

    cmap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    cmap.caption = "AgriVulnAI – Vulnerability score (low → high)"
    return cmap


def make_map(df: pd.DataFrame, output_html: str):
    # centru hartă
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    print(f"\n🌍 Map center: lat={center_lat:.3f}, lon={center_lon:.3f}")

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # colormap
    colormap = build_colormap(df)
    colormap.add_to(m)

    # layer puncte
    fg_points = folium.FeatureGroup(name="Vulnerability points", show=True)
    # layer heatmap
    fg_heatmap = folium.FeatureGroup(name="Vulnerability heatmap", show=False)

    heat_data = []

    for _, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        cls = float(row["pred_class"])
        vuln = float(row["vuln_score"])

        color = colormap(vuln)

        popup_html = (
            f"<b>Predicted class:</b> {cls:.1f}<br>"
            f"<b>Vulnerability score:</b> {vuln:.3f}<br>"
            f"<b>Lat:</b> {lat:.4f}<br>"
            f"<b>Lon:</b> {lon:.4f}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=3 + 4 * vuln,  # puțin mai mari la vulnerabilitate mare
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=0,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"class={cls:.1f}, vuln={vuln:.2f}",
        ).add_to(fg_points)

        heat_data.append([lat, lon, vuln])

    HeatMap(
        heat_data,
        radius=12,
        blur=16,
        max_zoom=8,
        min_opacity=0.3,
        name="Vulnerability heatmap",
    ).add_to(fg_heatmap)

    fg_points.add_to(m)
    fg_heatmap.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    m.save(output_html)
    print(f"\n✅ Map saved to: {output_html}")


def main():
    df = load_predictions(PREDICTIONS_CSV)
    df = add_vulnerability_score(df)
    make_map(df, OUTPUT_HTML)


if __name__ == "__main__":
    main()
