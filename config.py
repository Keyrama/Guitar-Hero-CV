# ============================================================
#  config.py — Parametri globali del progetto
# ============================================================

# --- Camera ---
CAMERA_INDEX = 0          # 0 = webcam default
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 60

# --- MediaPipe ---
MP_MAX_HANDS            = 1     # Una sola mano
MP_DETECTION_CONFIDENCE = 0.7   # Soglia rilevamento mano
MP_TRACKING_CONFIDENCE  = 0.7   # Soglia tracking landmark

# --- Gesture Recognizer ---
# Soglia angolo (gradi) sotto la quale un dito è considerato "piegato"
FINGER_BENT_THRESHOLD = 160.0

# Finestra temporale per lo smoothing (numero di frame)
SMOOTHING_WINDOW = 5

# Soglia confidence sotto la quale il punteggio è parziale
CONFIDENCE_PARTIAL_THRESHOLD = 0.6

# --- Mapping dita → tasti Guitar Hero ---
# 0=pollice, 1=indice, 2=medio, 3=anulare, 4=mignolo
FINGER_NAMES = {
    0: "Verde",
    1: "Rosso",
    2: "Giallo",
    3: "Blu",
    4: "Arancione"
}

FINGER_COLORS = {
    0: (0,   200,  0),    # Verde
    1: (0,   0,    200),  # Rosso  (BGR)
    2: (0,   200,  200),  # Giallo
    3: (200, 0,    0),    # Blu
    4: (0,   140,  255),  # Arancione
}

# --- Game Engine ---
GAME_BPM            = 90     # Velocità brano (battiti al minuto)
NOTE_SPEED          = 300    # Pixel/secondo di discesa delle note
HIT_WINDOW_MS       = 150    # Finestra temporale per colpire una nota (ms)
PERFECT_WINDOW_MS   = 50     # Finestra per "Perfect" (ms)

# Punteggi
SCORE_PERFECT  = 100
SCORE_GOOD     = 60
SCORE_PARTIAL  = 30   # Quando confidence < soglia
SCORE_MISS     = 0

# --- UI ---
WINDOW_TITLE    = "Guitar Hero — Hand Tracking CV"
UI_FONT_SIZE    = 28
LANE_WIDTH      = 100   # Larghezza di ogni corsia (5 corsie totali)
NOTE_HEIGHT     = 30
HIT_LINE_Y      = 600   # Posizione verticale della linea di colpo

# --- Paths ---
DATA_DIR        = "data/gestures"
MODELS_DIR      = "models"
ASSETS_DIR      = "assets/sounds"