# ============================================================
#  vision/trainer.py — Training SVM + MLP (16 classi)
#
#  Esegui con:  python vision/trainer.py
# ============================================================

import numpy as np
import pandas as pd
import pickle
import os
import sys
sys.path.insert(0, ".")

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from config import MODELS_DIR, CLASS_NAMES

DATASET_PATH = "data/gestures/gestures.csv"


def load_dataset():
    print(f"[INFO] Carico dataset da {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    X  = df.drop("class_id", axis=1).values.astype(np.float32)
    y  = df["class_id"].values.astype(int)
    print(f"[INFO] Campioni: {len(y)} | Classi: {sorted(np.unique(y))}")
    return X, y


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = (y_pred == y_test).mean()
    print(f"  Accuracy test set: {acc*100:.2f}%")

    cv_scores = cross_val_score(
        model,
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        cv=5
    )
    print(f"  Cross-val (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    print("\n  Report per classe:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(cm)

    return model


def save_model(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [INFO] Salvato: {path}")


def main():
    X, y = load_dataset()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("[INFO] Scaler salvato.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(y_train)} | Test: {len(y_test)}")

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    svm = train_and_evaluate("SVM (RBF kernel)", svm, X_train, X_test, y_train, y_test)
    save_model(svm, "svm")

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp = train_and_evaluate("MLP (256→128→64 ReLU)", mlp, X_train, X_test, y_train, y_test)
    save_model(mlp, "mlp")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    )
    rf = train_and_evaluate("Random Forest (200 alberi)", rf, X_train, X_test, y_train, y_test)
    save_model(rf, "rf")

    print("\n[INFO] Training completato. Modelli pronti in:", MODELS_DIR)


if __name__ == "__main__":
    main()