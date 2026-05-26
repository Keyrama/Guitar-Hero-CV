# ============================================================
#  config.py — Parametri globali del progetto
# ============================================================

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 60

# --- MediaPipe ---
MP_MAX_HANDS            = 1
MP_DETECTION_CONFIDENCE = 0.7
MP_TRACKING_CONFIDENCE  = 0.7

# --- Gesture Recognizer ---
# Finestra temporale per lo smoothing (numero di frame)
SMOOTHING_WINDOW = 5

# Soglia confidence sotto la quale il punteggio è parziale
CONFIDENCE_PARTIAL_THRESHOLD = 0.6

# --- Mapping classi → tasti Guitar Hero ---
# 4 dita: 1=indice, 2=medio, 3=anulare, 4=mignolo
# 16 classi totali (0=riposo, 1-15=combinazioni)

CLASS_NAMES = {
    0:  "Riposo",
    1:  "Indice",
    2:  "Medio",
    3:  "Anulare",
    4:  "Mignolo",
    5:  "Indice+Medio",
    6:  "Indice+Anulare",
    7:  "Indice+Mignolo",
    8:  "Medio+Anulare",
    9:  "Medio+Mignolo",
    10: "Anulare+Mignolo",
    11: "Indice+Medio+Anulare",
    12: "Indice+Medio+Mignolo",
    13: "Indice+Anulare+Mignolo",
    14: "Medio+Anulare+Mignolo",
    15: "Tutte e 4",
}

# Dita attive per ogni classe (1=indice,2=medio,3=anulare,4=mignolo)
CLASS_FINGERS = {
    0:  [],
    1:  [1],
    2:  [2],
    3:  [3],
    4:  [4],
    5:  [1, 2],
    6:  [1, 3],
    7:  [1, 4],
    8:  [2, 3],
    9:  [2, 4],
    10: [3, 4],
    11: [1, 2, 3],
    12: [1, 2, 4],
    13: [1, 3, 4],
    14: [2, 3, 4],
    15: [1, 2, 3, 4],
}

# Colori BGR per ogni dito
FINGER_COLORS = {
    1: (0,   200,  0),    # Verde  — Indice
    2: (0,   0,    200),  # Rosso  — Medio
    3: (0,   200,  200),  # Giallo — Anulare
    4: (200, 0,    0),    # Blu    — Mignolo
}

FINGER_NAMES = {
    1: "Verde  (Indice)",
    2: "Rosso  (Medio)",
    3: "Giallo (Anulare)",
    4: "Blu    (Mignolo)",
}

# --- Game Engine ---
GAME_BPM          = 90
NOTE_SPEED        = 300
HIT_WINDOW_MS     = 150
PERFECT_WINDOW_MS = 50

# Punteggi
SCORE_PERFECT = 100
SCORE_GOOD    = 60
SCORE_PARTIAL = 30
SCORE_MISS    = 0

# --- UI ---
WINDOW_TITLE = "Guitar Hero — Hand Tracking CV"
UI_FONT_SIZE = 28
LANE_WIDTH   = 100
NOTE_HEIGHT  = 30
HIT_LINE_Y   = 600

# --- Paths ---
DATA_DIR   = "data/gestures"
MODELS_DIR = "models"
ASSETS_DIR = "assets/sounds"