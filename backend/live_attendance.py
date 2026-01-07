import cv2
import pickle
import numpy as np
from datetime import datetime
from mtcnn import MTCNN
import os
import csv
import time
from scipy.spatial import distance as dist
from deepface import DeepFace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import firebase_admin
from firebase_admin import credentials, firestore

# ==============================
# Eye Aspect Ratio for Blink Detection
# ==============================
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ==============================
# Smile Detection
# ==============================
def is_smiling(mouth):
    left_corner = mouth[0]
    right_corner = mouth[6]
    top_lip = mouth[3]
    bottom_lip = mouth[9]
    width = dist.euclidean(left_corner, right_corner)
    height = dist.euclidean(top_lip, bottom_lip)
    ratio = width / height if height > 0 else 0
    return ratio > 2.0  # Adjust threshold as needed

# ==============================
# Load Face Encodings
# ==============================
ENCODING_PATH = os.path.join("data", "encodings.pkl")

if not os.path.exists(ENCODING_PATH):
    raise FileNotFoundError(f"Encodings file not found at {ENCODING_PATH}")

with open(ENCODING_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print("✅ Loaded encodings for:", set(known_names))

# ==============================
# Initialize MTCNN
# ==============================
detector = MTCNN()

# ==============================
# Initialize MediaPipe Face Mesh
# ==============================
# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ==============================
# Attendance Record
# ==============================
marked_names = set()
blink_counters = {}

# Initialize CSV if not exists
if not os.path.exists("data/attendance.csv"):
    with open("data/attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# ==============================
# Liveness States
# ==============================
liveness_state = {}  # name: {'state': 'idle', 'start_time': time, 'last_center': (x,y)}

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 1
HEAD_MOVE_THRESH = 50
SMILE_CONSEC_FRAMES = 5  # Consecutive frames smiling
TIMEOUT = 10  # seconds

# ==============================
# Firebase Initialization
# ==============================
try:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully")
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    db = None

# ==============================
# Start Camera
# ==============================
video = cv2.VideoCapture(0)

if not video.isOpened():
    raise RuntimeError("Could not start camera. Make sure it is accessible.")

print("🎥 Camera started. Press Q to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)

    # Detect faces
    faces = detector.detect_faces(rgb_frame)

    result = []
    if len(faces) > 0:
        try:
            result = DeepFace.represent(rgb_frame, model_name='VGG-Face', enforce_detection=True)
        except:
            result = []

    if len(faces) == 0:
        cv2.putText(frame, "No faces detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print(f"Faces detected: {len(faces)}, Results: {len(result)}")  # Debug
        for i, face in enumerate(faces):
            if i >= len(result):
                continue

            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)

            encoding = result[i]['embedding']

            # Compare with known encodings using cosine distance
            distances = [dist.cosine(encoding, known_enc) for known_enc in known_encodings]
            min_distance = min(distances)
            min_index = distances.index(min_distance)

            print(f"Min distance: {min_distance}")  # Debug

            name = "Unknown"
            if min_distance < 0.6:  # Threshold for match
                name = known_names[min_index]

                print(f"Recognized: {name}")  # Debug

                # Check for blink
                if results.face_landmarks:
                    face_landmarks = results.face_landmarks[0]  # Assuming one face
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    right_eye_indices = [362, 385, 387, 263, 373, 380]
                    left_eye = [(face_landmarks[idx].x * frame.shape[1], face_landmarks[idx].y * frame.shape[0]) for idx in left_eye_indices]
                    right_eye = [(face_landmarks[idx].x * frame.shape[1], face_landmarks[idx].y * frame.shape[0]) for idx in right_eye_indices]
                    ear_left = eye_aspect_ratio(left_eye)
                    ear_right = eye_aspect_ratio(right_eye)
                    ear = (ear_left + ear_right) / 2.0

                    if name not in blink_counters:
                        blink_counters[name] = 0

                    if ear < EYE_AR_THRESH:
                        blink_counters[name] += 1
                    else:
                        blink_counters[name] = 0

                    if blink_counters[name] >= EYE_AR_CONSEC_FRAMES:
                        # Mark attendance
                        if name not in marked_names:
                            marked_names.add(name)
                            time_now = datetime.now().strftime("%H:%M:%S")
                            date_now = datetime.now().strftime("%Y-%m-%d")
                            print(f"🟢 Attendance Marked: {name} at {time_now}")
                            
                            # Write to CSV
                            with open("data/attendance.csv", "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([name, date_now, time_now])
                                print(f"✅ Data written to CSV: {name}, {date_now}, {time_now}")
                            
                            # Upload to Firebase Firestore
                            if db:
                                try:
                                    doc_ref = db.collection('attendance').document()
                                    doc_ref.set({
                                        'name': name,
                                        'date': date_now,
                                        'time': time_now,
                                        'timestamp': firestore.SERVER_TIMESTAMP
                                    })
                                    print(f"☁️ Data uploaded to Firebase: {name}")
                                except Exception as e:
                                    print(f"❌ Firebase upload failed: {e}")
                            else:
                                print("⚠️ Firebase not initialized, skipping cloud upload")
                        instruction = "Attendance Marked!"
                    else:
                        instruction = f"Blink to confirm ({blink_counters[name]}/{EYE_AR_CONSEC_FRAMES})"
                else:
                    instruction = "Face recognized"

                if False:  # Disable liveness
                    pass
                    liveness_state[name] = {'state': 'idle', 'start_time': None, 'last_center': None, 'last_nose': None, 'blink_counter': 0, 'smile_counter': 0}

                # state = liveness_state[name]['state']
                # current_time = time.time()
                # current_center = (x + w // 2, y + h // 2)

                # # Get landmarks
                # landmarks = face_recognition.face_landmarks(face_img)
                current_nose = None
                # if landmarks:
                    # nose = landmarks[0]['nose_bridge'][3]  # Middle of nose bridge
                    # current_nose = (nose[0], nose[1])

                    # left_eye = landmarks[0]['left_eye']
                    # right_eye = landmarks[0]['right_eye']
                    # mouth = landmarks[0]['top_lip'] + landmarks[0]['bottom_lip']  # Combine for mouth
                    # ear_left = eye_aspect_ratio(left_eye)
                    # ear_right = eye_aspect_ratio(right_eye)
                    # ear = (ear_left + ear_right) / 2.0

                    # if ear < EYE_AR_THRESH:
                        # liveness_state[name]['blink_counter'] += 1
                    # else:
                        # liveness_state[name]['blink_counter'] = 0

                    # Check for smile
                    # if is_smiling(mouth):
                        # liveness_state[name]['smile_counter'] += 1
                    # else:
                        # liveness_state[name]['smile_counter'] = 0

                # if state == 'idle':
                    # liveness_state[name]['state'] = 'blink'
                    # liveness_state[name]['start_time'] = current_time
                    # instruction = "Blink your eyes"
                # elif state == 'blink':
                    # if liveness_state[name]['blink_counter'] >= EYE_AR_CONSEC_FRAMES:
                        # liveness_state[name]['state'] = 'move'
                        # liveness_state[name]['start_time'] = current_time
                        # liveness_state[name]['last_center'] = current_center
                        # liveness_state[name]['last_nose'] = current_nose
                        # instruction = "Move your head left/right"
                    # elif current_time - liveness_state[name]['start_time'] > TIMEOUT:
                        # liveness_state[name]['state'] = 'idle'
                        # instruction = "Timeout. Try again."
                    # else:
                        # instruction = "Blink your eyes"
                # elif state == 'move':
                    # last_pos = liveness_state[name]['last_nose'] if liveness_state[name]['last_nose'] else liveness_state[name]['last_center']
                    # current_pos = current_nose if current_nose else current_center
                    # if last_pos and current_pos:
                        # move_dist = dist.euclidean(last_pos, current_pos)
                        # if move_dist > HEAD_MOVE_THRESH:
                            # liveness_state[name]['state'] = 'smile'
                            # liveness_state[name]['start_time'] = current_time
                            # instruction = "Smile"
                        # elif current_time - liveness_state[name]['start_time'] > TIMEOUT:
                            # liveness_state[name]['state'] = 'idle'
                            # instruction = "Timeout. Try again."
                        # else:
                            # instruction = "Move your head left/right"
                    # else:
                        # instruction = "Move your head left/right"
                # elif state == 'smile':
                    # if liveness_state[name]['smile_counter'] >= SMILE_CONSEC_FRAMES:
                        # # Mark attendance
                        # if name not in marked_names:
                            # marked_names.add(name)
                            # time_now = datetime.now().strftime("%H:%M:%S")
                            # date_now = datetime.now().strftime("%Y-%m-%d")
                            # print(f"🟢 Attendance Marked: {name} at {time_now}")
                            
                            # # Write to CSV
                            # with open("backend/data/attendance.csv", "a", newline="") as f:
                                # writer = csv.writer(f)
                                # writer.writerow([name, date_now, time_now])
                        # liveness_state[name]['state'] = 'done'
                        # instruction = "Attendance Marked!"
                    # elif current_time - liveness_state[name]['start_time'] > TIMEOUT:
                        # liveness_state[name]['state'] = 'idle'
                        # instruction = "Timeout. Try again."
                    # else:
                        # instruction = "Smile"
                # else:
                    # instruction = "Verified"

                # # Display instruction
                # cv2.putText(frame, f"{name}: {instruction}", (x, y-30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                instruction = ""

            # Draw bounding box and name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Fraud-Proof Attendance System", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# Cleanup
# ==============================
video.release()
cv2.destroyAllWindows()
print("🛑 Camera stopped.")
