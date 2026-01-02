import os
import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime

# ------------------------
# Path setup
# ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENC_FILE = os.path.join(BASE_DIR, "data", "encodings.pkl")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "data", "attendance.csv")

# ------------------------
# Load known face encodings
# ------------------------
with open(ENC_FILE, "rb") as f:
    known_encodings_dict = pickle.load(f)

known_face_encodings = []
known_face_names = []

for name, enc_list in known_encodings_dict.items():
    for enc in enc_list:
        # Ensure numpy array of float64
        known_face_encodings.append(np.array(enc, dtype=np.float64))
        known_face_names.append(name)

print(f"✅ Known faces loaded: {list(set(known_face_names))}")

# ------------------------
# Attendance storage
# ------------------------
attendance = {}  # {name: timestamp}

# Load previous attendance if CSV exists
if os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            name, date, time = line.strip().split(",")
            attendance[name] = f"{date} {time}"

# ------------------------
# Webcam setup
# ------------------------
cap = cv2.VideoCapture(0)
print("📷 Press Q to quit the attendance system")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Cannot read from camera")
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        name = "Unknown"
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Mark attendance if not already marked
            if name not in attendance:
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")
                attendance[name] = f"{date_str} {time_str}"
                print(f"✅ Attendance marked for {name} at {date_str} {time_str}")

        # Draw rectangle and name
        top, right, bottom, left = [v*4 for v in face_location]  # scale back
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------
# Save attendance to CSV
# ------------------------
with open(ATTENDANCE_FILE, "w") as f:
    f.write("Name,Date,Time\n")
    for name, dt in attendance.items():
        date, time = dt.split()
        f.write(f"{name},{date},{time}\n")

print(f"🎉 Attendance saved at {ATTENDANCE_FILE}")

cap.release()
cv2.destroyAllWindows()
