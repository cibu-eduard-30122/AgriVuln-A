import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend fără ecran (Colab / server)
import matplotlib.pyplot as plt
import shap


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
ROOT_DIR = "/content/drive/MyDrive/AgriVulnAI"
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
ARTIFACT_PATH = os.path.join(RESULTS_DIR, "shap_values.joblib")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "shap_plots")


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------------------
# Load SHAP artifacts
# ------------------------------------------------------------------------------
def load_shap_artifacts(artifact_path: str):
    """
    Expected content in shap_values.joblib:
      {
          "X_test_sample": <array or DataFrame, shape (n_samples, n_features)>,
          "shap_values": <np.array, shape (n_samples, n_features, n_classes)>,
          (optional) "feature_names": list[str],
          (optional) "class_names": list[str]
      }
    """
    data = joblib.load(artifact_path)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a dict in {artifact_path}, got {type(data)}. "
            "Please save shap_values.joblib as a dict."
        )

    if "shap_values" not in data:
        raise KeyError("Key 'shap_values' not found in shap_values.joblib")

    shap_values = data["shap_values"]

    # Alegem X_test_sample fără să folosim 'or' pe DataFrame (care dă eroare)
    if "X_test_sample" in data:
        X_test_sample = data["X_test_sample"]
    elif "X_test" in data:
        X_test_sample = data["X_test"]
    elif "X" in data:
        X_test_sample = data["X"]
    else:
        raise KeyError(
            "No 'X_test_sample', 'X_test' or 'X' found in shap_values.joblib. "
            "Please include the test feature matrix in the artifact."
        )

    # Feature names
    if "feature_names" in data:
        feature_names = data["feature_names"]
    elif hasattr(X_test_sample, "columns"):
        feature_names = list(X_test_sample.columns)
    else:
        n_features = shap_values.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Class names
    if shap_values.ndim == 3:
        n_classes = shap_values.shape[2]
    else:
        n_classes = 1

    if "class_names" in data:
        class_names = data["class_names"]
    else:
        class_names = [f"class_{i}" for i in range(n_classes)]

    return X_test_sample, shap_values, feature_names, class_names


# ------------------------------------------------------------------------------
# Plot functions
# ------------------------------------------------------------------------------
def plot_shap_beeswarm(X, shap_values, class_idx: int, class_name: str):
    """
    Beeswarm SHAP pentru o singură clasă.
    shap_values: shape (n_samples, n_features, n_classes)
    """
    shap_values_class = shap_values[:, :, class_idx]  # (n_samples, n_features)

    features_input = X  # poate fi DataFrame sau np.array

    print(f"  -> Beeswarm pentru clasa {class_idx} ({class_name})")
    shap.summary_plot(
        shap_values_class,
        features_input,
        feature_names=getattr(X, "columns", None),
        show=False
    )
    plt.tight_layout()
    out_path = os.path.join(
        OUTPUT_DIR,
        f"shap_beeswarm_class_{class_idx}_{class_name}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"     Salvat: {out_path}")


def plot_shap_bar_global(X, shap_values, class_idx: int, class_name: str):
    """
    Bar-plot SHAP global (mean |SHAP|) pentru o singură clasă.
    """
    shap_values_class = shap_values[:, :, class_idx]

    print(f"  -> Bar summary pentru clasa {class_idx} ({class_name})")
    shap.summary_plot(
        shap_values_class,
        X,
        feature_names=getattr(X, "columns", None),
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    out_path = os.path.join(
        OUTPUT_DIR,
        f"shap_bar_global_class_{class_idx}_{class_name}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"     Salvat: {out_path}")


def plot_shap_heatmap(shap_matrix, feature_names, class_names):
    """
    Heatmap (features × classes) pe baza shap_matrix: shape (n_features, n_classes).
    """
    print("  -> Heatmap shap_matrix (features × classes)")
    plt.figure(figsize=(1 + 0.4 * len(class_names), 0.4 * len(feature_names)))
    im = plt.imshow(shap_matrix, aspect="auto")

    plt.yticks(range(len(feature_names)), feature_names)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")

    cbar = plt.colorbar(im)
    cbar.set_label("mean |SHAP value|")

    plt.title("Global SHAP importance (mean |SHAP|) per feature & class")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "shap_heatmap_features_x_classes.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"     Salvat: {out_path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    print(ROOT_DIR)
    X_test_sample, shap_values, feature_names, class_names = load_shap_artifacts(
        ARTIFACT_PATH
    )
    print(f"✅ Loaded SHAP artifacts from {ARTIFACT_PATH}")

    # Ne asigurăm că numărul de sample-uri din X și shap_values se potrivește
    n_samples_shap = shap_values.shape[0]
    if hasattr(X_test_sample, "shape"):
        n_samples_X = X_test_sample.shape[0]
    else:
        n_samples_X = len(X_test_sample)

    if n_samples_X > n_samples_shap:
        X_test_sample = X_test_sample[:n_samples_shap]

    print(f"- X_test_sample shape: {X_test_sample.shape}")
    print(f"- shap_values shape:   {shap_values.shape}")

    if shap_values.ndim != 3:
        raise ValueError(
            f"Expected shap_values with 3 dims (samples, features, classes), got shape {shap_values.shape}"
        )

    n_samples, n_features, n_classes = shap_values.shape
    print(f"  -> n_samples: {n_samples}, n_features: {n_features}, n_classes: {n_classes}")

    # shap_matrix: mean(|SHAP|) peste samples -> (n_features, n_classes)
    shap_matrix = np.mean(np.abs(shap_values), axis=0)
    print(f"- shap_matrix final shape: {shap_matrix.shape}")

    # HEATMAP global (features × classes)
    plot_shap_heatmap(shap_matrix, feature_names, class_names)

    # Pentru fiecare clasă:
    for class_idx in range(n_classes):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        print(f"\n=== Plots pentru clasa {class_idx} ({class_name}) ===")
        plot_shap_beeswarm(X_test_sample, shap_values, class_idx, class_name)
        plot_shap_bar_global(X_test_sample, shap_values, class_idx, class_name)

    print("\n✅ Gata. Toate figurile SHAP au fost salvate în:")
    print(f"   {OUTPUT_DIR}")


if __name__ == "__main__":
    shap.initjs()
    main()
