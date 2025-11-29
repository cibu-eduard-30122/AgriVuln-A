#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgriVulnAI – Kenya Vulnerability Dashboard (Plotly-only version)
- Fără folium / streamlit_folium.
- Mapă interactivă cu Plotly (scatter + heatmap).
- Integrează Vulnerability Index v2 + clustere + interpretare.
"""

from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Config & paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

CSV_V2 = DATA_DIR / "ground_truth_with_predictions_v2.csv"
CSV_V1 = DATA_DIR / "ground_truth_with_predictions.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_cluster_text(cluster_id: int) -> str:
    mapping = {
        0: "0 – low",
        1: "1 – medium",
        2: "2 – high",
        3: "3 – very high",
    }
    return mapping.get(int(cluster_id), str(cluster_id))


def build_hover_text(row: pd.Series) -> str:
    """
    Text compact pentru hover Plotly.
    Nu mai afișăm 'nan', ci 'NA' (+ iconițe).
    """

    def fmt(v, dec: int = 3) -> str:
        if pd.isna(v):
            return "NA"
        try:
            return f"{float(v):.{dec}f}"
        except Exception:
            return str(v)

    parts: List[str] = []
    parts.append(f"<b>Class (model)</b>: {int(row['pred_class'])}")
    parts.append(f"<b>Cluster v2</b>: {get_cluster_text(int(row['vuln_cluster_v2']))}")
    parts.append(f"<b>Vuln index v2</b>: {fmt(row['vuln_index_v2'], 3)}")
    parts.append(f"<b>Lat</b>: {fmt(row['lat'], 3)}, <b>Lon</b>: {fmt(row['lon'], 3)}")

    # sat + climate features (dacă există coloana, o arătăm)
    if "NDVI_2023-03-01" in row.index:
        parts.append(f"🌿 <b>NDVI</b>: {fmt(row['NDVI_2023-03-01'], 3)}")
    if "EVI_2023-03-01" in row.index:
        parts.append(f"🌱 <b>EVI</b>: {fmt(row['EVI_2023-03-01'], 3)}")
    if "CHIRPS_precip_2023-03-01" in row.index:
        parts.append(f"💧 <b>CHIRPS precip</b>: {fmt(row['CHIRPS_precip_2023-03-01'], 1)}")
    if "PM25_2023-03-01" in row.index:
        parts.append(f"💨 <b>PM25</b>: {fmt(row['PM25_2023-03-01'], 3)}")
    if "Water_occurrence" in row.index:
        parts.append(f"💦 <b>Water occurrence</b>: {fmt(row['Water_occurrence'], 1)}")
    if "WorldPop_2020" in row.index:
        parts.append(f"👥 <b>WorldPop 2020</b>: {fmt(row['WorldPop_2020'], 0)}")

    if "pred_confidence" in row.index:
        parts.append(f"✅ <b>Prediction confidence</b>: {fmt(row['pred_confidence'], 3)}")

    # interpretare doar dacă nu e string gol / NaN
    if (
        "interpretation" in row.index
        and isinstance(row["interpretation"], str)
        and row["interpretation"].strip()
    ):
        parts.append(f"📌 <b>Interpretation</b>: {row['interpretation']}")

    return "<br>".join(parts)


def summarize_subset(df: pd.DataFrame) -> Dict[str, str]:
    """
    Text scurt 'story mode' pentru subsetul filtrat.
    """
    out: Dict[str, str] = {}

    if df.empty:
        out["main"] = "No data for current filters."
        return out

    mean_v = df["vuln_index_v2"].mean()
    min_v = df["vuln_index_v2"].min()
    max_v = df["vuln_index_v2"].max()
    out["stats"] = (
        f"Mean vulnerability v2: **{mean_v:.3f}** | "
        f"Min: **{min_v:.3f}** | Max: **{max_v:.3f}**"
    )

    # distribuție pe clustere
    clusters = df["vuln_cluster_v2"].value_counts().sort_index()
    parts = []
    for c in range(4):
        cnt = int(clusters.get(c, 0))
        parts.append(f"{get_cluster_text(c)}: **{cnt}** points")
    out["clusters"] = " | ".join(parts)

    # ceva interpretare simplă bazată pe NDVI / CHIRPS
    msg_bits: List[str] = []
    if "NDVI_2023-03-01" in df.columns:
        ndvi_mean = df["NDVI_2023-03-01"].mean()
        if ndvi_mean < 0.2:
            msg_bits.append("vegetation is **stressed** (low NDVI)")
        elif ndvi_mean > 0.6:
            msg_bits.append("vegetation is **healthy** (high NDVI)")
        else:
            msg_bits.append("vegetation is at **moderate** levels")

    if "CHIRPS_precip_2023-03-01" in df.columns:
        precip = df["CHIRPS_precip_2023-03-01"].mean()
        if precip > 150:
            msg_bits.append("precipitation is **high** (intense rainfall episodes)")
        elif precip < 30:
            msg_bits.append("precipitation is **low** (dry conditions)")
        else:
            msg_bits.append("precipitation is at **normal** levels")

    if msg_bits:
        out["drivers"] = "In this filtered subset, " + "; ".join(msg_bits) + "."
    else:
        out["drivers"] = ""

    return out


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Încarcă CSV-ul cu predicții.
    Preferăm v2 dacă există, altfel fallback la v1.
    """
    if CSV_V2.exists():
        csv_path = CSV_V2
    else:
        csv_path = CSV_V1

    df = pd.read_csv(csv_path)

    # asigură coloanele esențiale
    if "vuln_index_v2" not in df.columns:
        # dacă nu avem v2, normalizăm pred_class ca fallback (0-1)
        df["vuln_index_v2"] = (df["pred_class"] - df["pred_class"].min()) / (
            df["pred_class"].max() - df["pred_class"].min()
        )

    if "vuln_cluster_v2" not in df.columns:
        df["vuln_cluster_v2"] = 0

    if "pred_confidence" not in df.columns:
        df["pred_confidence"] = 1.0

    # interpretare – dacă nu există, punem string gol
    if "interpretation" not in df.columns:
        df["interpretation"] = ""

    return df


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="AgriVulnAI – Kenya Vulnerability Dashboard",
        page_icon="🌍",
        layout="wide",
    )

    df = load_data()

    st.sidebar.title("Filters")

    # --- filtrare după class (model original) ---
    min_class = int(df["pred_class"].min())
    max_class = int(df["pred_class"].max())
    class_range = st.sidebar.slider(
        "Predicted class range (original model)",
        min_value=min_class,
        max_value=max_class,
        value=(min_class, max_class),
        step=1,
    )

    # --- filtru clustere v2 ---
    cluster_options = [0, 1, 2, 3]
    cluster_labels = [get_cluster_text(c) for c in cluster_options]
    default_clusters = cluster_options  # toate selectate
    selected_labels = st.sidebar.multiselect(
        "Vulnerability cluster v2 (0=low, 3=very high)",
        options=cluster_labels,
        default=cluster_labels,
    )
    selected_clusters = [cluster_options[cluster_labels.index(lbl)] for lbl in selected_labels]

    # --- filtru pe încredere model ---
    min_conf = float(
        st.sidebar.slider(
            "Min. prediction confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Map display")
    show_points = st.sidebar.checkbox("Show points", value=True)
    show_heatmap = st.sidebar.checkbox("Show heatmap", value=True)

    point_size_factor = st.sidebar.slider(
        "Point size (relative)", 0.1, 2.0, 1.0, 0.05
    )
    heatmap_intensity = st.sidebar.slider(
        "Heatmap intensity", 0.1, 2.0, 1.0, 0.05
    )

    basemap_style_name = st.sidebar.selectbox(
        "Basemap style", ["Dark", "Light", "Outdoors"]
    )
    basemap_style_mapbox = {
        "Dark": "carto-darkmatter",
        "Light": "carto-positron",
        "Outdoors": "open-street-map",
    }[basemap_style_name]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Culoare și mărime: verde = vulnerabilitate scăzută, galben = medie, roșu = ridicată. "
        "Heatmap-ul evidențiază agregarea zonelor cu vulnerabilitate mare."
    )

    # -----------------------------------------------------------------------------
    # Main layout
    # -----------------------------------------------------------------------------
    st.title("AgriVulnAI – Kenya Vulnerability Dashboard")

    st.markdown(
        """
Dashboard interactiv pentru a explora predicțiile de vulnerabilitate agricolă în Kenya.

- **Map**: puncte colorate după Vulnerability Index v2 + heatmap.
- **SHAP**: importanța indicatorilor satelitari și socio-economici (model LightGBM pe puncte).
- **Data**: tabel cu datele filtrate, exportabil în CSV.
"""
    )

    tab_map, tab_shap, tab_data = st.tabs(["🗺 Map", "📊 SHAP", "📄 Data"])

    # -----------------------------------------------------------------------------
    # MAP TAB
    # -----------------------------------------------------------------------------
    with tab_map:
        # filtrare dataframe
        mask = (
            (df["pred_class"] >= class_range[0])
            & (df["pred_class"] <= class_range[1])
            & (df["vuln_cluster_v2"].isin(selected_clusters))
            & (df["pred_confidence"] >= min_conf)
        )
        df_filt = df[mask].copy()
        df_filt = df_filt.dropna(subset=["lat", "lon", "vuln_index_v2"])

        if df_filt.empty:
            st.warning("No points match current filters.")
            return

        center_lat = float(df_filt["lat"].mean())
        center_lon = float(df_filt["lon"].mean())

        # stats panel + map
        col_stats, col_map = st.columns([1, 2])

        with col_stats:
            st.subheader("Stats")
            st.metric("Filtered points", f"{len(df_filt):,}")
            st.metric("Mean vulnerability index v2", f"{df_filt['vuln_index_v2'].mean():.3f}")
            st.metric("Min vulnerability index v2", f"{df_filt['vuln_index_v2'].min():.3f}")
            st.metric("Max vulnerability index v2", f"{df_filt['vuln_index_v2'].max():.3f}")

        with col_map:
            st.subheader("Vulnerability Map (Index v2)")

            # build hover text
            customdata = df_filt.apply(build_hover_text, axis=1)

            fig = go.Figure()

            # heatmap
            if show_heatmap:
                fig.add_trace(
                    go.Densitymapbox(
                        lat=df_filt["lat"],
                        lon=df_filt["lon"],
                        z=df_filt["vuln_index_v2"],
                        radius=20 * heatmap_intensity,
                        coloraxis="coloraxis",
                        opacity=0.6,
                        hoverinfo="skip",
                    )
                )

            # points
            if show_points:
                fig.add_trace(
                    go.Scattermapbox(
                        lat=df_filt["lat"],
                        lon=df_filt["lon"],
                        mode="markers",
                        marker=dict(
                            size=6 + 6 * point_size_factor,
                            color=df_filt["vuln_index_v2"],
                            coloraxis="coloraxis",
                            opacity=0.9,
                        ),
                        customdata=customdata,
                        hovertemplate="%{customdata}<extra></extra>",
                    )
                )

            fig.update_layout(
                mapbox=dict(
                    style=basemap_style_mapbox,
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=6,
                ),
                coloraxis=dict(
                    colorscale="Turbo",
                    cmin=0.0,
                    cmax=1.0,
                    colorbar=dict(
                        title="Vulnerability index v2 (0 = low, 1 = high)",
                    ),
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=650,
            )

            # noul API: width în loc de use_container_width
            st.plotly_chart(fig, width="stretch")

        # Story mode / summary
        st.markdown("---")
        st.subheader("📖 Story mode – summary for current filters")

        summary = summarize_subset(df_filt)
        st.markdown(summary.get("stats", ""))
        st.markdown(summary.get("clusters", ""))
        if summary.get("drivers"):
            st.markdown(summary["drivers"])

    # -----------------------------------------------------------------------------
    # SHAP TAB (simplu – imagine dacă există)
    # -----------------------------------------------------------------------------
    with tab_shap:
        st.subheader("SHAP – feature importance")
        shap_png = BASE_DIR / "results" / "shap_plots" / "shap_summary.png"
        if shap_png.exists():
            st.image(str(shap_png), caption="SHAP summary plot (model LightGBM)")
        else:
            st.info("SHAP summary image not found. Place it in `results/shap_plots/shap_summary.png`.")

    # -----------------------------------------------------------------------------
    # DATA TAB
    # -----------------------------------------------------------------------------
    with tab_data:
        st.subheader("Filtered data table")

        mask = (
            (df["pred_class"] >= class_range[0])
            & (df["pred_class"] <= class_range[1])
            & (df["vuln_cluster_v2"].isin(selected_clusters))
            & (df["pred_confidence"] >= min_conf)
        )
        df_filt = df[mask].copy().reset_index(drop=True)

        st.dataframe(df_filt, height=500)

        csv_bytes = df_filt.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data as CSV",
            data=csv_bytes,
            file_name="agrivulnai_filtered_points.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
