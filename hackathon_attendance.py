import cv2
import face_recognition
import pickle
import csv
import os
from datetime import datetime

# ---------------- CONFIG ----------------
ENCODINGS_PATH = "backend/data/encodings.pkl"
ATTENDANCE_FILE = "backend/data/attendance.csv"
BLINK_FRAMES_REQUIRED = 2   # simple blink threshold
# ----------------------------------------

# Load known encodings
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Attendance tracking
marked_today = set()

# Initialize camera ONCE
cap = cv2.VideoCapture(0)
print("📷 Camera started. Press 'q' to quit.")

blink_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    # ---------- BACKGROUND BLUR ----------
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    output = blurred.copy()

    for (top, right, bottom, left), face_encoding in zip(locations, encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

        # Restore original face (no blur)
        output[top:bottom, left:right] = frame[top:bottom, left:right]

        # ---------- DRAW FACE BOX ----------
        cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            output,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ---------- SIMPLE BLINK LOGIC ----------
        blink_counter += 1
        if blink_counter >= BLINK_FRAMES_REQUIRED and name != "Unknown":
            today = datetime.now().strftime("%Y-%m-%d")

            if name not in marked_today:
                time_now = datetime.now().strftime("%H:%M:%S")

                os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)
                file_exists = os.path.isfile(ATTENDANCE_FILE)

                with open(ATTENDANCE_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Name", "Date", "Time"])
                    writer.writerow([name, today, time_now])

                marked_today.add(name)
                print(f"📝 Attendance marked for {name} at {time_now}")

    cv2.imshow("Fraud-Proof Attendance", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup (IMPORTANT)
cap.release()
cv2.destroyAllWindows()
print("🛑 Camera stopped.")





