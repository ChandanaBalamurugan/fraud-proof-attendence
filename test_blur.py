import cv2
import face_recognition
import pickle
import numpy as np
import os

# 📁 Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "backend", "data", "encodings.pickle")

# 🔓 Load encodings
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print("✅ Encodings loaded:", len(known_encodings))

# 🎥 Camera
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("❌ Camera not accessible")
    exit()

print("🎥 Camera started. Press Q to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # ✅ Ensure proper format (CRITICAL FIX)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = np.ascontiguousarray(frame, dtype=np.uint8)

    # 🔍 Face detection
    face_locations = face_recognition.face_locations(frame, model="hog")
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Convert back for OpenCV display
    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=0.5
        )

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        else:
            # 🔒 Blur unknown face
            face = display_frame[top:bottom, left:right]
            face = cv2.GaussianBlur(face, (99, 99), 30)
            display_frame[top:bottom, left:right] = face

        # 🟦 Draw box
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            display_frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Fraud-Proof Attendance", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()


