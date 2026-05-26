# ============================================================
#  data/collector.py — Raccolta dataset gesti
#
#  Esegui con:  python data/collector.py
#
#  Per ogni dito (0-4) registra N campioni del gesto
#  "quel dito piegato, gli altri estesi".
#  I dati vengono salvati in data/gestures/gestures.csv
# ============================================================

import cv2
import csv
import sys
import time
import os
sys.path.insert(0, ".")

from vision.hand_tracker import HandTracker
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    FINGER_NAMES, FINGER_COLORS,
)

# --- Parametri raccolta ---
SAMPLES_PER_CLASS = 200   # campioni per ogni classe
COUNTDOWN_SEC     = 3     # secondi di countdown prima di registrare
OUTPUT_FILE       = "data/gestures/gestures.csv"

# Classi: 0-4 = un solo dito piegato, 5 = nessun dito piegato (riposo)
CLASS_DESCRIPTIONS = {
    0: "VERDE  — piega solo il POLLICE",
    1: "ROSSO  — piega solo l'INDICE",
    2: "GIALLO — piega solo il MEDIO",
    3: "BLU    — piega solo l'ANULARE",
    4: "ARANCIONE — piega solo il MIGNOLO",
    5: "RIPOSO — mano aperta, nessun dito piegato",
}


def draw_ui(frame, class_id, phase, count, total, countdown=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    color = FINGER_COLORS.get(class_id, (200, 200, 200))
    desc  = CLASS_DESCRIPTIONS[class_id]

    cv2.putText(frame, f"Classe {class_id}: {desc}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if phase == "countdown" and countdown is not None:
        msg = f"Preparati... {countdown}"
        cv2.putText(frame, msg, (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    elif phase == "recording":
        msg = f"Registrazione: {count}/{total}"
        cv2.putText(frame, msg, (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif phase == "done":
        cv2.putText(frame, "Classe completata! Premi SPAZIO per continuare.",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif phase == "wait":
        cv2.putText(frame, "Premi SPAZIO per iniziare questa classe.",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def main():
    os.makedirs("data/gestures", exist_ok=True)

    tracker = HandTracker()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERRORE] Webcam non trovata.")
        return

    # Apri CSV in append (così puoi riprendere se interrompi)
    file_exists = os.path.exists(OUTPUT_FILE)
    csvfile = open(OUTPUT_FILE, "a", newline="")
    writer  = csv.writer(csvfile)

    # Header solo se file nuovo
    if not file_exists:
        header = ["class_id"] + [f"f{i}" for i in range(63)]
        writer.writerow(header)

    print(f"[INFO] Dataset salvato in: {OUTPUT_FILE}")
    print(f"[INFO] {SAMPLES_PER_CLASS} campioni per classe, {len(CLASS_DESCRIPTIONS)} classi")
    print("[INFO] Premi SPAZIO per iniziare ogni classe, Q per uscire.\n")

    for class_id, desc in CLASS_DESCRIPTIONS.items():
        print(f"\n--- Classe {class_id}: {desc} ---")
        phase   = "wait"
        count   = 0
        countdown_start = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            landmarks = tracker.process(frame)
            if landmarks:
                tracker.draw(frame, landmarks)

            # --- Stato: attesa input utente ---
            if phase == "wait":
                draw_ui(frame, class_id, "wait", count, SAMPLES_PER_CLASS)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    phase = "countdown"
                    countdown_start = time.time()
                elif key == ord('q'):
                    csvfile.close()
                    cap.release()
                    tracker.release()
                    cv2.destroyAllWindows()
                    return

            # --- Stato: countdown ---
            elif phase == "countdown":
                elapsed   = time.time() - countdown_start
                remaining = int(COUNTDOWN_SEC - elapsed) + 1
                draw_ui(frame, class_id, "countdown", count, SAMPLES_PER_CLASS, remaining)
                cv2.waitKey(1)
                if elapsed >= COUNTDOWN_SEC:
                    phase = "recording"

            # --- Stato: registrazione ---
            elif phase == "recording":
                draw_ui(frame, class_id, "recording", count, SAMPLES_PER_CLASS)
                cv2.waitKey(1)

                if landmarks and count < SAMPLES_PER_CLASS:
                    features = tracker.landmarks_to_array(landmarks).tolist()
                    writer.writerow([class_id] + features)
                    csvfile.flush()
                    count += 1

                if count >= SAMPLES_PER_CLASS:
                    phase = "done"

            # --- Stato: classe completata ---
            elif phase == "done":
                draw_ui(frame, class_id, "done", count, SAMPLES_PER_CLASS)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break  # passa alla classe successiva
                elif key == ord('q'):
                    csvfile.close()
                    cap.release()
                    tracker.release()
                    cv2.destroyAllWindows()
                    return

            cv2.imshow("Raccolta Dataset Gesti", frame)

    csvfile.close()
    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Dataset completo salvato in {OUTPUT_FILE}")
    print(f"[INFO] Totale campioni: {len(CLASS_DESCRIPTIONS) * SAMPLES_PER_CLASS}")


if __name__ == "__main__":
    main()