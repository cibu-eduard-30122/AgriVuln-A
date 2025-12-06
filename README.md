AgriVulnAI is an end-to-end geospatial machine learning system developed to estimate agricultural vulnerability in Kenya by integrating:
Sentinel-2 vegetation indices (NDVI, EVI)
CHIRPS precipitation
PM2.5 air quality
CAMS aerosol indicators
Hydrological layers (Water Occurrence)
Socio-economic data (WorldPop 2020)
A LightGBM classifier trained on vulnerability ground truth
The project produces:
A composite Vulnerability Index v2
A 4-level vulnerability clustering (Low → Very High)
An interactive geospatial dashboard for exploration
Feature importance explanations via SHAP
The pipeline is optimized for research, policy-making, and early-stage climate-risk product development.



Project Structure:
AgriVuln-A/
│
├── dashboard_app.py                # Main Streamlit dashboard
├── compute_vulnerability_index_v2.py
├── data_pipeline.py
├── modeling.py                     # LightGBM model training
├── preprocess.py                   # Data cleaning & prep
│
├── data/
│   └── processed/
│       ├── ground_truth_with_predictions.csv
│       ├── ground_truth_with_predictions_v2.csv
│       ├── features_ground_truth.csv
│       └── features_ground_truth_with_extra.csv
│
├── results/
│   ├── metrics.csv
│   ├── confusion_matrix.csv
│   ├── shap_plots/
│   │     └── shap_summary.png     # Add this manually
│   └── map_points_predictions.png
│
├── figures/                        # Interactive HTML maps
├── requirements.txt
└── README.md
1. Machine Learning Results

Model used: LightGBM, trained on 4,925 labeled points.

✔ Overall performance
Metric	Value
Accuracy	0.965
F1-macro	0.908
Confusion Matrix:
[[ 17   0  10   1   0   0]
 [  0   1   0   0   0   0]
 [  8   0 857   3   0   2]
 [  1   0   9  51   0   0]
 [  0   0   0   0   2   0]
 [  0   0   0   0   0  21]]
Interpretation:
Class 4 is the dominant and best-predicted class
0 and 7 show moderate confusion → expected due to overlapping spectral patterns
Rare classes (1, 9, 10) achieve perfect or near-perfect precision due to clear signatures

2. Vulnerability Index v2
The improved index integrates:
Vegetation stress (NDVI, EVI)
Hydrological stability (Water Occurrence)
Precipitation anomalies (CHIRPS)
Air pollution (PM2.5 + CAMS)
Local exposure (WorldPop density)

Distribution stats (0–1 scale):
mean: 0.518
std: 0.179
min: 0.00
max: 1.00
The new index is smoother, more interpretable and responds correctly to both climate and population pressure.

3. Vulnerability Clustering (K=4)
K-Means produced four distinct vulnerability groups:
| Cluster | Description | Avg. Index | Count |
| ------- | ----------- | ---------- | ----- |
| **0**   | Low         | 0.405      | 2556  |
| **1**   | Medium      | 0.584      | 375   |
| **2**   | High        | 0.595      | 1007  |
| **3**   | Very High   | 0.706      | 987   |
These clusters power the dashboard filters + insights panel.

4. Recent Improvements
 KNN Imputation for Missing Data
Missing values (NDVI, EVI, CAMS, Water Occurrence, WorldPop) were filled using KNN Imputer (k=5).
Benefits:
Removes all NaNs
Captures local spatial similarity
Improves cluster separation
Enhances interpretability

 Streamlit Dashboard Rewrite (Plotly-only)
Removed Folium → replaced with modern scattermap + densitymap
Improved hover cards with icons
Heatmap and point size controls
Story Mode: automatic textual interpretation
Integrated SHAP summary tab

5. Interactive Dashboard Features
The dashboard displays:
 Geospatial Map
Colored by vulnerability index
Heatmap aggregating high-risk regions
Click / hover to inspect satellite metrics

 Statistics Panel
Shows current filtered subset:
Mean / min / max vulnerability
Cluster distribution
NDVI & precipitation interpretation

SHAP Explanations
Add the file:results/shap_plots/shap_summary.png
to view feature importance in the dashboard.
