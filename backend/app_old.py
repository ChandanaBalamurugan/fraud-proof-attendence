import cv2
import face_recognition
import pickle
import os
from datetime import datetime

ENCODINGS_PATH = "backend/data/encodings.pickle"
ATTENDANCE_PATH = "backend/data/attendance.csv"

# Load encodings
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Initialize webcam
video = cv2.VideoCapture(0)

marked_names = set()

print("📸 Live Attendance Started... Press Q to stop")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, box in zip(encodings, boxes):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

            if name not in marked_names:
                marked_names.add(name)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(ATTENDANCE_PATH, "a") as f:
                    f.write(f"{name},{now}\n")

                print(f"✅ Attendance marked for {name}")

        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Fraud-Proof Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
