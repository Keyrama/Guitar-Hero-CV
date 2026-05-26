import cv2
import csv
import sys
import time
import os
sys.path.insert(0, ".")

from vision.hand_tracker import HandTracker
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

SAMPLES_PER_CLASS = 200
COUNTDOWN_SEC     = 3
OUTPUT_FILE       = "data/gestures/gestures.csv"

CLASS_INSTRUCTIONS = {
    0:  "Mano aperta - nessun dito piegato",
    1:  "Piega solo l'INDICE",
    2:  "Piega solo il MEDIO",
    3:  "Piega solo l'ANULARE",
    4:  "Piega solo il MIGNOLO",
    5:  "Piega INDICE + MEDIO",
    6:  "Piega INDICE + ANULARE",
    7:  "Piega INDICE + MIGNOLO",
    8:  "Piega MEDIO + ANULARE",
    9:  "Piega MEDIO + MIGNOLO",
    10: "Piega ANULARE + MIGNOLO",
    11: "Piega INDICE + MEDIO + ANULARE",
    12: "Piega INDICE + MEDIO + MIGNOLO",
    13: "Piega INDICE + ANULARE + MIGNOLO",
    14: "Piega MEDIO + ANULARE + MIGNOLO",
    15: "Piega TUTTE E 4 le dita",
}


def main():
    os.makedirs("data/gestures", exist_ok=True)

    tracker = HandTracker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERRORE] Webcam non trovata.")
        return

    file_exists = os.path.exists(OUTPUT_FILE)
    csvfile = open(OUTPUT_FILE, "a", newline="")
    writer  = csv.writer(csvfile)
    if not file_exists:
        header = ["class_id"] + [f"f{i}" for i in range(63)]
        writer.writerow(header)

    print(f"[INFO] 16 classi x {SAMPLES_PER_CLASS} campioni = {16 * SAMPLES_PER_CLASS} totali")
    print("[INFO] SPAZIO = inizia/avanza | Q = esci\n")

    for class_id in range(16):
        print(f"\n--- Classe {class_id}: {CLASS_INSTRUCTIONS[class_id]} ---")
        phase           = "wait"
        count           = 0
        countdown_start = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            landmarks = tracker.process(frame)
            if landmarks:
                tracker.draw(frame, landmarks)

            # --- UI ---
            istr = CLASS_INSTRUCTIONS[class_id]
            cv2.putText(frame, f"Classe {class_id}: {istr}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            if phase == "wait":
                cv2.putText(frame, "Premi SPAZIO per iniziare",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            elif phase == "countdown":
                elapsed   = time.time() - countdown_start
                remaining = int(COUNTDOWN_SEC - elapsed) + 1
                cv2.putText(frame, f"Preparati... {remaining}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                if elapsed >= COUNTDOWN_SEC:
                    phase = "recording"

            elif phase == "recording":
                cv2.putText(frame, f"Registrazione: {count}/{SAMPLES_PER_CLASS}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                if landmarks and count < SAMPLES_PER_CLASS:
                    features = tracker.landmarks_to_array(landmarks).tolist()
                    writer.writerow([class_id] + features)
                    csvfile.flush()
                    count += 1
                if count >= SAMPLES_PER_CLASS:
                    phase = "done"

            elif phase == "done":
                cv2.putText(frame, "Completata! Premi SPAZIO per continuare.",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow("Raccolta Dataset", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                csvfile.close()
                cap.release()
                tracker.release()
                cv2.destroyAllWindows()
                return

            if key == ord(' '):
                if phase == "wait":
                    phase = "countdown"
                    countdown_start = time.time()
                elif phase == "done":
                    break

    csvfile.close()
    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Dataset completo: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()