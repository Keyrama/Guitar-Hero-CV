# ============================================================
#  test_tracker.py — Test rapido HandTracker + webcam
#
#  Esegui con:  python test_tracker.py
#  Premi 'q' per uscire
# ============================================================

import cv2
import sys
import time
sys.path.insert(0, ".")

from vision.hand_tracker import HandTracker
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    FINGER_NAMES, FINGER_COLORS, FINGER_BENT_THRESHOLD
)


def main():
    tracker = HandTracker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERRORE] Webcam non trovata. Controlla CAMERA_INDEX in config.py")
        return

    print("[INFO] Webcam aperta. Premi 'q' per uscire.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRORE] Frame non letto.")
            break

        # Specchia il frame (effetto specchio, più naturale)
        frame = cv2.flip(frame, 1)

        # --- Processa la mano ---
        landmarks = tracker.process(frame)

        if landmarks:
            # Disegna landmark
            tracker.draw(frame, landmarks)

            # Calcola angoli e stato dita
            angles = tracker.finger_angles(landmarks)

            # Mostra stato dita nell'angolo in alto a sinistra
            for finger_id, angle in angles.items():
                is_bent  = angle < FINGER_BENT_THRESHOLD
                stato    = "PIEGATO" if is_bent else "ESTESO "
                color    = FINGER_COLORS[finger_id]
                nome     = FINGER_NAMES[finger_id]
                testo    = f"{nome}: {stato}  ({angle:.1f} deg)"

                cv2.putText(
                    frame, testo,
                    (10, 40 + finger_id * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2
                )
        else:
            cv2.putText(
                frame, "Nessuna mano rilevata",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2
            )

        # --- FPS counter ---
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (FRAME_WIDTH - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (200, 200, 200), 2
        )

        cv2.imshow("HandTracker Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print("[INFO] Test terminato.")


if __name__ == "__main__":
    main()
