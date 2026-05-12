"""
ASL Real-time Inference
=======================
Preprocessing identico al layer Preprocess del notebook di training
(soluzione 1 posto Kaggle "Google Isolated Sign Language Recognition").

Input modello : (1, 384, 708)  — gia' preprocessato
Pipeline      : MediaPipe Holistic -> frame (543, 3) -> Preprocess -> (384, 708)
"""

import os, zipfile, threading
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ==============================================================================
#  CONFIGURAZIONE
# ==============================================================================

ZIP_PATH    = "asl_pure_tf_blackbox.zip"
EXTRACT_DIR = "_savedmodel_extracted"

SEQUENCE_LENGTH      = 384
ROWS_PER_FRAME       = 543   # landmark totali per frame (face+lhand+pose+rhand)
CONFIDENCE_THRESHOLD = 0.4

# ==============================================================================
#  LANDMARK INDICES  (dal notebook di training — NON modificare)
# ==============================================================================

LIP = [
    0,  61, 185,  40,  39,  37, 267, 269, 270, 409,
  291, 146,  91, 181,  84,  17, 314, 405, 321, 375,
   78, 191,  80,  81,  82,  13, 312, 311, 310, 415,
   95,  88, 178,  87,  14, 317, 402, 318, 324, 308,
]  # 40 landmark labbra

LHAND = list(range(468, 489))   # 21 — indici globali mano sinistra
RHAND = list(range(522, 543))   # 21 — indici globali mano destra

NOSE = [1, 2, 98, 327]          # 4

REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173]          # 16

LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398]          # 16

# Ordine di concatenazione: LIP + LHAND + RHAND + NOSE + REYE + LEYE
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
# 40+21+21+4+16+16 = 118 landmark × 2 coord (x,y) = 236 feature base
# 236 × 3 (base + dx + dx2) = 708 ✓

NUM_NODES = len(POINT_LANDMARKS)   # 118

_check = (len(LIP) + len(LHAND) + len(RHAND) + len(NOSE) + len(REYE) + len(LEYE))
assert _check == 118, f"Conteggio landmark errato: {_check}"

# ==============================================================================
#  MAPPA CLASSI (250 segni ASL — ordine da sign_to_prediction_index_map.json)
#
#  IMPORTANTE: inserisci qui il JSON dal dataset Kaggle:
#  https://www.kaggle.com/competitions/asl-signs/data
#  Caricalo con:
#      import json
#      with open('sign_to_prediction_index_map.json') as f:
#          m = json.load(f)
#      LABELS = [k for k, v in sorted(m.items(), key=lambda x: x[1])]
#
#  Finche' non la carichi vengono mostrati i numeri di classe.
# ==============================================================================

SIGN_MAP_PATH = "sign_to_prediction_index_map.json"
LABELS: list[str] = []

if os.path.isfile(SIGN_MAP_PATH):
    import json
    with open(SIGN_MAP_PATH) as f:
        _m = json.load(f)
    LABELS = [k for k, v in sorted(_m.items(), key=lambda x: x[1])]
    print(f"[labels] caricate {len(LABELS)} classi da {SIGN_MAP_PATH}")

def label_of(idx: int) -> str:
    return LABELS[idx] if LABELS and idx < len(LABELS) else f"#{idx}"

# ==============================================================================
#  ESTRAZIONE FRAME  (543, 3) — NaN dove il landmark non e' rilevato
# ==============================================================================

def extract_frame(results) -> np.ndarray:
    """
    Costruisce un array (543, 3) con x,y,z di ogni landmark.
    Struttura globale (identica al dataset Kaggle):
        0   – 467  : face (MediaPipe face_landmarks)
        468 – 488  : left hand
        489 – 521  : pose
        522 – 542  : right hand
    """
    frame = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)

    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            frame[i, 0] = lm.x
            frame[i, 1] = lm.y
            frame[i, 2] = lm.z

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            frame[468 + i, 0] = lm.x
            frame[468 + i, 1] = lm.y
            frame[468 + i, 2] = lm.z

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            frame[489 + i, 0] = lm.x
            frame[489 + i, 1] = lm.y
            frame[489 + i, 2] = lm.z

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            frame[522 + i, 0] = lm.x
            frame[522 + i, 1] = lm.y
            frame[522 + i, 2] = lm.z

    return frame

# ==============================================================================
#  PREPROCESSING  — replica esatta del layer Preprocess del notebook
# ==============================================================================

def _nan_mean(x, axis, keepdims=True):
    """Media NaN-aware (equivale a tf_nan_mean del notebook)."""
    with np.errstate(all='ignore'):
        return np.nanmean(x, axis=axis, keepdims=keepdims)

def _nan_std_centered(x, center, axis, keepdims=True):
    """Deviazione standard NaN-aware centrata su un valore esterno."""
    d = x - center
    with np.errstate(all='ignore'):
        var = np.nanmean(d * d, axis=axis, keepdims=keepdims)
    return np.sqrt(var)


def preprocess_sequence(frames: deque) -> np.ndarray:
    """
    Trasforma la deque di SEQUENCE_LENGTH frame (543,3) in un array (384, 708).

    Pipeline identica al layer Preprocess del notebook:
      1. Media del landmark 17 su tutti i frame (punto di riferimento stabile)
      2. Selezione POINT_LANDMARKS: (T, 543, 3) -> (T, 118, 3)
      3. Z-score rispetto al punto 17 (media) e alla dispersione globale (std)
      4. Tronca a MAX_LEN frame
      5. Mantieni solo x, y  (drop z)
      6. dx  = x[t+1] - x[t], padding zero alla fine
      7. dx2 = x[t+2] - x[t], padding zero agli ultimi 2 frame
      8. Concatena [x, dx, dx2] -> (T, 708)
      9. NaN -> 0
    """
    # (T, 543, 3) con batch dim
    x = np.array(frames, dtype=np.float32)[np.newaxis]   # (1, T, 543, 3)
    T = x.shape[1]

    # 1. Media del landmark 17 su tutti i frame e le coordinate
    lm17 = x[:, :, 17:18, :]          # (1, T, 1, 3)
    mean = _nan_mean(lm17, axis=(1, 2), keepdims=True)   # (1, 1, 1, 3)
    mean = np.where(np.isnan(mean), np.float32(0.5), mean)

    # 2. Seleziona POINT_LANDMARKS
    x = x[:, :, POINT_LANDMARKS, :]   # (1, T, 118, 3)

    # 3. Std centrata su mean, calcolata su T e su tutti i 118 landmark
    std = _nan_std_centered(x, mean, axis=(1, 2), keepdims=True)  # (1,1,1,3)
    std = np.where((std == 0) | np.isnan(std), np.float32(1.0), std)

    x = (x - mean) / std              # (1, T, 118, 3)

    # 4. Mantieni solo x, y
    x = x[..., :2]                    # (1, T, 118, 2)

    # 5. Derivate in avanti (come nel notebook con tf.pad)
    #    dx[t]  = x[t+1] - x[t],  dx[T-1]   = 0
    #    dx2[t] = x[t+2] - x[t],  dx2[T-2:] = 0
    dx  = np.zeros_like(x)
    dx2 = np.zeros_like(x)
    if T > 1:
        dx[:, :-1]  = x[:, 1:]  - x[:, :-1]
    if T > 2:
        dx2[:, :-2] = x[:, 2:]  - x[:, :-2]

    # 6. Reshape e concatena -> (1, T, 708)
    N = NUM_NODES * 2   # 236
    x_r   = x.reshape(1, T, N)
    dx_r  = dx.reshape(1, T, N)
    dx2_r = dx2.reshape(1, T, N)
    result = np.concatenate([x_r, dx_r, dx2_r], axis=-1)   # (1, T, 708)

    # 7. NaN -> 0
    result = np.where(np.isnan(result), np.float32(0.0), result)

    return result[0].astype(np.float32)   # (T, 708) = (384, 708)

# ==============================================================================
#  CARICAMENTO MODELLO
# ==============================================================================

def load_savedmodel(zip_path: str, extract_dir: str):
    pb_path = None
    if os.path.isdir(extract_dir):
        for root, _, files in os.walk(extract_dir):
            if "saved_model.pb" in files:
                pb_path = root
                break

    if pb_path is None:
        print(f"[model] estrazione {zip_path} -> {extract_dir}/")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        for root, _, files in os.walk(extract_dir):
            if "saved_model.pb" in files:
                pb_path = root
                break

    print(f"[model] caricamento da: {pb_path}")
    model   = tf.saved_model.load(pb_path)
    sig     = model.signatures["serving_default"]
    in_key  = list(sig.structured_input_signature[1].keys())[0]
    out_key = list(sig.structured_outputs.keys())[0]
    print(f"[model] input='{in_key}'  output='{out_key}'")

    def infer(arr: np.ndarray) -> np.ndarray:
        tensor = tf.constant(arr[np.newaxis], dtype=tf.float32)  # (1,384,708)
        return sig(**{in_key: tensor})[out_key].numpy()[0]        # (250,)

    return infer

# ==============================================================================
#  THREAD DI INFERENZA
# ==============================================================================

class InferenceWorker(threading.Thread):
    def __init__(self, infer_fn):
        super().__init__(daemon=True, name="InferenceWorker")
        self._infer   = infer_fn
        self._pending = None
        self._lock    = threading.Lock()
        self._event   = threading.Event()
        self._running = True
        self.prediction = "Raccolta dati..."
        self.confidence = 0.0

    def submit(self, arr: np.ndarray):
        with self._lock:
            self._pending = arr
        self._event.set()

    def stop(self):
        self._running = False
        self._event.set()

    def run(self):
        while self._running:
            self._event.wait()
            self._event.clear()
            if not self._running:
                break
            with self._lock:
                arr, self._pending = self._pending, None
            if arr is None:
                continue
            try:
                logits  = self._infer(arr)
                probs   = tf.nn.softmax(logits).numpy()
                top_idx = int(np.argmax(probs))
                top_p   = float(probs[top_idx])
                self.prediction = label_of(top_idx) if top_p >= CONFIDENCE_THRESHOLD else "???"
                self.confidence = top_p
            except Exception as exc:
                print(f"[InferenceWorker] {exc}")

# ==============================================================================
#  HUD
# ==============================================================================

def draw_hud(frame, text, confidence, queue_len):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    color = (0, 220, 80) if confidence >= CONFIDENCE_THRESHOLD else (60, 180, 220)
    cv2.putText(frame, f"Segno: {text}   {confidence:.0%}",
                (12, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)
    bar_h = 18
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    fill  = int(w * queue_len / SEQUENCE_LENGTH)
    col   = (0, 200, 80) if queue_len == SEQUENCE_LENGTH else (0, 130, 200)
    cv2.rectangle(frame, (0, h - bar_h), (fill, h), col, -1)
    cv2.putText(frame, f"Buffer {queue_len}/{SEQUENCE_LENGTH}  [q per uscire]",
                (6, h - 3), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1, cv2.LINE_AA)

# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    infer_fn = load_savedmodel(ZIP_PATH, EXTRACT_DIR)
    worker   = InferenceWorker(infer_fn)
    worker.start()

    mp_holistic = mp.solutions.holistic
    mp_draw     = mp.solutions.drawing_utils
    spec_face   = mp_draw.DrawingSpec(color=(100, 130, 20), thickness=1, circle_radius=1)
    spec_pose   = mp_draw.DrawingSpec(color=(30,  80, 200), thickness=2, circle_radius=3)
    spec_hand   = mp_draw.DrawingSpec(color=(200,  30, 100), thickness=2, circle_radius=3)

    sequence: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"[camera] avviata. Buffer: {SEQUENCE_LENGTH} frame "
          f"(~{SEQUENCE_LENGTH/30:.1f}s a 30fps). Premi 'q' per uscire.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            if results.face_landmarks:
                mp_draw.draw_landmarks(frame, results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS, spec_face, spec_face)
            mp_draw.draw_landmarks(frame, results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS, spec_pose, spec_pose)
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS, spec_hand, spec_hand)
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS, spec_hand, spec_hand)

            sequence.append(extract_frame(results))

            if len(sequence) == SEQUENCE_LENGTH:
                worker.submit(preprocess_sequence(sequence))

            draw_hud(frame, worker.prediction, worker.confidence, len(sequence))
            cv2.imshow("ASL Real-time Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    worker.stop()
    worker.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
