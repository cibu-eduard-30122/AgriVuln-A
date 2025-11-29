# modeling.py
#
# Folosește features_ground_truth_with_extra.csv (sau features_ground_truth.csv)
# pentru a antrena un model LightGBM multiclass.
# Salvează:
#   - results/lightgbm_model.joblib
#   - results/shap_values.joblib
#   - results/metrics.csv
#   - results/confusion_matrix.csv

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import lightgbm as lgb
import shap

from config import PROCESSED_DIR, RESULTS_DIR


def load_features_dataframe():
    """
    Încearcă să folosească mai întâi features_ground_truth_with_extra.csv
    (cu WorldPop, apă, PM2.5). Dacă nu există, cade înapoi pe
    features_ground_truth.csv.
    """
    path_extra = PROCESSED_DIR / "features_ground_truth_with_extra.csv"
    path_base = PROCESSED_DIR / "features_ground_truth.csv"

    if path_extra.exists():
        print(f"📂 Loading extended features from {path_extra}")
        df = pd.read_csv(path_extra)
        used_path = path_extra
    elif path_base.exists():
        print(f"📂 Loading base features from {path_base}")
        df = pd.read_csv(path_base)
        used_path = path_base
    else:
        raise FileNotFoundError(
            "Nu am găsit nici features_ground_truth_with_extra.csv, "
            "nici features_ground_truth.csv în data/processed/"
        )

    print(f"✅ DataFrame shape: {df.shape}")
    return df, used_path


def prepare_X_y(df: pd.DataFrame, min_samples_per_class: int = 5):
    """
    Extrage X (features numerice) și y (label).
    Filtrează clasele foarte rare (< min_samples_per_class).
    """

    if "label" not in df.columns:
        raise ValueError(
            f"Nu am găsit coloana 'label' în features CSV. "
            f"Coloane disponibile: {list(df.columns)}"
        )

    # y = ținta
    y = df["label"].astype(int)

    # X = toate coloanele numerice, exceptând labelul
    num_df = df.select_dtypes(include=[np.number])
    if "label" in num_df.columns:
        X = num_df.drop(columns=["label"])
    else:
        X = num_df

    print(f"DEBUG: X shape (before filtering classes): {X.shape}")
    print("DEBUG: primele coloane și statistici:")
    print(X.describe().T[["mean", "std", "min", "max"]].head(20))

    print("DEBUG: număr de valori unice pe coloană (primele 20):")
    print(X.nunique().head(20))

    # distribuția de clase înainte de filtrare
    class_counts = y.value_counts()
    print("Class distribution BEFORE filtering (label -> count):")
    print(class_counts)

    # păstrăm doar clasele cu suficient suport
    valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
    print(f"Keeping classes with at least {min_samples_per_class} samples: {valid_classes}")

    mask = y.isin(valid_classes)
    X_filtered = X.loc[mask].reset_index(drop=True)
    y_filtered = y.loc[mask].reset_index(drop=True)

    # distribuția de clase după filtrare
    print("Class distribution AFTER filtering (label -> count):")
    print(y_filtered.value_counts())

    print(
        f"Number of samples after filtering: {len(y_filtered)}, "
        f"number of classes: {y_filtered.nunique()}"
    )

    return X_filtered, y_filtered


def train_lightgbm_classifier(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Antrenează un LGBMClassifier multiclass pe X, y.
    Returnează modelul și X_test, y_test, preds.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    num_classes = y.nunique()

    print("Training LightGBM classifier...")
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy = {acc:.3f}")
    print(f"F1-macro = {f1_macro:.3f}")

    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    print("Confusion matrix:")
    print(cm)

    report = classification_report(y_test, y_pred)
    print("Classification report:")
    print(report)

    return clf, X_test, y_test, y_pred, acc, f1_macro, cm, report


def save_metrics_and_artifacts(
    model,
    X_test,
    y_test,
    y_pred,
    acc,
    f1_macro,
    cm,
    report: str,
    features_path,
):
    """
    Salvează modelul, SHAP values, metrics.csv și confusion_matrix.csv.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS_DIR / "lightgbm_model.joblib"
    shap_path = RESULTS_DIR / "shap_values.joblib"
    metrics_path = RESULTS_DIR / "metrics.csv"
    cm_path = RESULTS_DIR / "confusion_matrix.csv"

    # model
    joblib.dump(model, model_path)

    # SHAP (pentru câteva sute de eșantioane, ca să nu fie uriaș fișierul)
    print("Computing SHAP values (may take a bit)...")
    sample_size = min(1000, len(X_test))
    X_shap = X_test.iloc[:sample_size]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    joblib.dump(
        {"X_test_sample": X_shap, "shap_values": shap_values},
        shap_path,
    )

    # metrics csv
    metrics = {
        "metric": ["accuracy", "f1_macro", "n_test_samples", "features_source"],
        "value": [acc, f1_macro, len(y_test), str(features_path)],
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)

    # confusion matrix
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(cm_path, index=False)

    print(f"Model saved to {model_path}")
    print(f"SHAP values saved to {shap_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Confusion matrix saved to {cm_path}")


def main():
    os.chdir("/content/drive/MyDrive/AgriVulnAI")
    print(os.getcwd())

    # 1) încărcăm features CSV (cu extra indices, dacă există)
    df, used_path = load_features_dataframe()

    # 2) construim X, y + filtrăm clasele foarte rare
    X, y = prepare_X_y(df, min_samples_per_class=5)

    # 3) antrenăm LightGBM
    model, X_test, y_test, y_pred, acc, f1_macro, cm, report = train_lightgbm_classifier(X, y)

    # 4) salvăm model, SHAP și metrici
    save_metrics_and_artifacts(
        model,
        X_test,
        y_test,
        y_pred,
        acc,
        f1_macro,
        cm,
        report,
        used_path,
    )


if __name__ == "__main__":
    main()
