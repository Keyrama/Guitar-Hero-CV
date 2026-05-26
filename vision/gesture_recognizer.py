# ============================================================
#  vision/gesture_recognizer.py — ML Recognizer (16 classi)
# ============================================================

import numpy as np
import os
import pickle
from config import MODELS_DIR, CLASS_NAMES, CONFIDENCE_PARTIAL_THRESHOLD


class MLRecognizer:
    """
    Classificatore ML (SVM o MLP) trainato sul dataset a 16 classi.
    Ritorna (class_id, confidence).
    """

    def __init__(self, model_name: str = "svm"):
        self.model_name = model_name
        self.model      = None
        self.scaler     = None
        self._load()

    def _load(self):
        model_path  = os.path.join(MODELS_DIR, f"{self.model_name}.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modello non trovato: {model_path}\n"
                f"Esegui prima: python vision/trainer.py"
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        print(f"[INFO] Modello caricato: {model_path}")

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """
        features: array flat 63 elementi (output di landmarks_to_array)
        Ritorna (class_id, confidence)
        """
        x          = self.scaler.transform(features.reshape(1, -1))
        class_id   = int(self.model.predict(x)[0])
        proba      = self.model.predict_proba(x)[0]
        confidence = float(proba[class_id])
        return class_id, confidence

    def is_partial(self, confidence: float) -> bool:
        return confidence < CONFIDENCE_PARTIAL_THRESHOLD

    def class_name(self, class_id: int) -> str:
        return CLASS_NAMES.get(class_id, "Sconosciuto")