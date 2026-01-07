import cv2
import pickle
import numpy as np
from deepface import DeepFace

ENCODINGS_PATH = "backend/data/encodings.pkl"

with open(ENCODINGS_PATH, "rb") as f:
    known_encodings = pickle.load(f)

cap = cv2.VideoCapture(0)
THRESHOLD = 10

print("🎥 Camera started... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        embedding = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        name = "Unknown"
        min_dist = float("inf")

        for person, embeds in known_encodings.items():
            for e in embeds:
                dist = np.linalg.norm(np.array(embedding) - np.array(e))
                if dist < min_dist:
                    min_dist = dist
                    name = person

        if min_dist > THRESHOLD:
            name = "Unknown"

        cv2.putText(frame, name, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except:
        pass

    cv2.imshow("Fraud-Proof Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
