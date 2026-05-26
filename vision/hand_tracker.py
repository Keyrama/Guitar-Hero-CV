# ============================================================
#  vision/hand_tracker.py — Wrapper MediaPipe Hands 0.10+
# ============================================================

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
from config import (
    MP_MAX_HANDS,
    MP_DETECTION_CONFIDENCE,
    MP_TRACKING_CONFIDENCE,
)

# Indici landmark per ogni dito [MCP, PIP, DIP, TIP]
FINGER_LANDMARK_INDICES = {
    1: [5,  6,  7,  8],   # Indice
    2: [9,  10, 11, 12],  # Medio
    3: [13, 14, 15, 16],  # Anulare
    4: [17, 18, 19, 20],  # Mignolo
}

WRIST_INDEX = 0
MODEL_PATH  = "hand_landmarker.task"
MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def _download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Download modello MediaPipe in corso...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Modello scaricato: {MODEL_PATH}")


class HandTracker:
    def __init__(self):
        _download_model()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MP_MAX_HANDS,
            min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._last_landmarks = None
        self._frame_timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> list[dict] | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._frame_timestamp_ms += 33  # ~30fps timestamp incrementale
        result = self._detector.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.hand_landmarks:
            self._last_landmarks = None
            return None

        hand = result.hand_landmarks[0]
        landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]
        self._last_landmarks = landmarks
        return landmarks

    def finger_angles(self, landmarks: list[dict]) -> dict[int, float]:
        angles = {}
        for finger_id, indices in FINGER_LANDMARK_INDICES.items():
            mcp, pip, dip, _ = indices
            A = np.array([landmarks[mcp]["x"], landmarks[mcp]["y"]])
            B = np.array([landmarks[pip]["x"], landmarks[pip]["y"]])
            C = np.array([landmarks[dip]["x"], landmarks[dip]["y"]])

            v1 = A - B
            v2 = C - B
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles[finger_id] = np.degrees(np.arccos(cos_angle))
        return angles

    def landmarks_to_array(self, landmarks: list[dict]) -> np.ndarray:
        wrist = np.array([landmarks[WRIST_INDEX]["x"],
                          landmarks[WRIST_INDEX]["y"],
                          landmarks[WRIST_INDEX]["z"]])
        feature_vector = []
        for lm in landmarks:
            pt = np.array([lm["x"], lm["y"], lm["z"]])
            feature_vector.extend(pt - wrist)
        return np.array(feature_vector, dtype=np.float32)

    def draw(self, frame_bgr: np.ndarray, landmarks: list[dict]) -> None:
        if landmarks is None:
            return
        h, w, _ = frame_bgr.shape

        # Connessioni mano definite manualmente (stesse di MediaPipe)
        CONNECTIONS = [
            (0,5),(5,6),(6,7),(7,8),           # Indice
            (0,9),(9,10),(10,11),(11,12),      # Medio
            (0,13),(13,14),(14,15),(15,16),    # Anulare
            (0,17),(17,18),(18,19),(19,20),    # Mignolo
            (5,9),(9,13),(13,17),              # Palmo
        ]

        pts = {i: (int(lm["x"] * w), int(lm["y"] * h)) for i, lm in enumerate(landmarks)}

        for start, end in CONNECTIONS:
            cv2.line(frame_bgr, pts[start], pts[end], (80, 80, 80), 2)

        for idx, (cx, cy) in pts.items():
            if idx in (1,2,3,4):
                continue
            cv2.circle(frame_bgr, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame_bgr, str(idx), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    def release(self):
        self._detector.close()