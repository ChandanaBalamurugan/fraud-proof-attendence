import os
import cv2
import pickle
import numpy as np
import time
from collections import defaultdict
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import face_recognition
import firebase_admin
from firebase_admin import credentials, firestore
from liveness_detection import get_detector

# ------------------------ PATH SETUP ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENC_FILE = os.path.normpath(os.path.join(BASE_DIR, "data", "encodings.pkl"))
ATTENDANCE_FILE = os.path.normpath(os.path.join(BASE_DIR, "data", "attendance.csv"))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ROOT_ATTENDANCE_FILE = os.path.normpath(os.path.join(PROJECT_ROOT, "data", "attendence.csv"))

os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)
os.makedirs(os.path.dirname(ROOT_ATTENDANCE_FILE), exist_ok=True)

TOLERANCE = 0.45  # Face distance threshold

# ------------------------ LOGGING ------------------------
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "attendance.log")

logger = logging.getLogger("attendance")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=1024*1024, backupCount=3, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

logger.info("Logging initialized. Log file: %s", LOG_FILE)

# ------------------------ FIREBASE INIT ------------------------
try:
    cred = credentials.Certificate(os.path.join(BASE_DIR, "serviceAccountKey.json"))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("✅ Firebase Connected Successfully")
except Exception as e:
    logger.exception("❌ Firebase initialization failed: %s", e)
    raise

# ------------------------ LOAD KNOWN FACE ENCODINGS ------------------------
with open(ENC_FILE, "rb") as f:
    known_encodings_dict = pickle.load(f)

known_face_encodings = []
known_face_names = []
for name, enc_list in known_encodings_dict.items():
    arr = np.array(enc_list, dtype=np.float64)
    centroid = np.mean(arr, axis=0) if arr.ndim > 1 else arr
    known_face_encodings.append(centroid)
    known_face_names.append(name)

logger.info("✅ Known faces loaded: %s", list(set(known_face_names)))
print(f"✅ Known faces loaded: {list(set(known_face_names))}")

# ------------------------ LIVENESS DETECTOR ------------------------
liveness_detector = get_detector(threshold=0.5)
logger.info("🔍 Liveness detector initialized")
print("🔍 Initializing liveness detector...")

# ------------------------ ATTENDANCE TRACKING ------------------------
attendance = {}     # {name: timestamp string} for session
session_seen = set()  # Avoid duplicate marks in same run
recent_detections = defaultdict(list)
DETECTION_WINDOW_SEC = 4.0
REQUIRED_HITS = 3

# Load previous attendance (informational only)
previous_names = set()
if os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        data_lines = lines[1:] if lines[0].lower().startswith("name") else lines
        for line in data_lines:
            parts = line.split(",")
            if len(parts) == 3:
                previous_names.add(parts[0])

# ------------------------ HELPER FUNCTIONS ------------------------
def _ensure_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as fh:
            fh.write("Name,Date,Time\n")

def _append_row(path, name, date_str, time_str):
    _ensure_header(path)
    with open(path, "a") as fh:
        fh.write(f"{name},{date_str},{time_str}\n")
    logger.info("Appended attendance row to %s: %s,%s,%s", path, name, date_str, time_str)

def _upload_to_firebase(name, date_str, time_str):
    doc_id = f"{name}_{date_str}_{time_str}"
    try:
        db.collection("attendance").document(doc_id).set({
            "Name": name,
            "Date": date_str,
            "Time": time_str,
            "Timestamp": firestore.SERVER_TIMESTAMP
        })
        logger.info("✅ Uploaded to Firebase: %s", doc_id)
        print(f"✅ Uploaded to Firebase: {doc_id}")
    except Exception as e:
        logger.exception("❌ Firebase upload failed for %s: %s", doc_id, e)
        print(f"❌ Firebase upload failed for {doc_id}: {e}")

# ------------------------ WEBCAM LOOP ------------------------
cap = cv2.VideoCapture(0)
print("📷 Press Q to quit the attendance system")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Cannot read from camera")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        is_live, liveness_conf = liveness_detector.is_live(small_frame, face_location)
        if not is_live:
            top, right, bottom, left = [v*4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
            cv2.putText(frame, "SPOOF", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            continue

        # Match face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "Unknown"
        if len(face_distances) > 0:
            best_idx = int(np.argmin(face_distances))
            if face_distances[best_idx] <= TOLERANCE:
                candidate = known_face_names[best_idx]
                now_ts = time.time()
                recent_detections[candidate].append(now_ts)
                recent_detections[candidate] = [t for t in recent_detections[candidate] if now_ts - t <= DETECTION_WINDOW_SEC]
                if len(recent_detections[candidate]) >= REQUIRED_HITS:
                    name = candidate

        # Mark attendance
        if name != "Unknown" and name not in session_seen:
            has_blinks = liveness_detector.has_confirmed_blinks()
            has_motion = liveness_detector._has_sufficient_motion()
            if has_blinks or has_motion:
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")
                session_seen.add(name)
                attendance[name] = f"{date_str} {time_str}"

                print(f"✅ Attendance marked for {name} at {date_str} {time_str}")
                logger.info("Attendance marked for %s at %s %s", name, date_str, time_str)

                # Append CSV
                _append_row(ATTENDANCE_FILE, name, date_str, time_str)
                _append_row(ROOT_ATTENDANCE_FILE, name, date_str, time_str)

                # Upload to Firebase
                _upload_to_firebase(name, date_str, time_str)

        # Draw rectangle and label
        top, right, bottom, left = [v*4 for v in face_location]
        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance System (Press Q to Quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🎉 Attendance session finished. Files updated and Firebase synced.")
logger.info("Attendance session finished. Files updated and Firebase synced.")
