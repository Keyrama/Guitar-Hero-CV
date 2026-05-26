# ============================================================
#  test_tracker.py — Test HandTracker + webcam
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
    FINGER_NAMES, FINGER_COLORS,
)


def main():
    tracker = HandTracker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERRORE] Webcam non trovata.")
        return

    print("[INFO] Webcam aperta. Premi 'q' per uscire.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        landmarks = tracker.process(frame)

        if landmarks:
            tracker.draw(frame, landmarks)

            # Mostra le 4 dita attive (ignora pollice)
            for i, (finger_id, nome) in enumerate(FINGER_NAMES.items()):
                color = FINGER_COLORS[finger_id]
                cv2.putText(
                    frame, nome,
                    (10, 40 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    color, 2
                )
        else:
            cv2.putText(frame, "Nessuna mano rilevata",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (FRAME_WIDTH - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 2)

        cv2.imshow("HandTracker Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print("[INFO] Test terminato.")


if __name__ == "__main__":
    main()