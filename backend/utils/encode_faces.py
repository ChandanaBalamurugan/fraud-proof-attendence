import os
import pickle
from deepface import DeepFace
import cv2

DATASET_DIR = "backend/data/faces"
ENCODINGS_FILE = "backend/data/encodings.pkl"

known_encodings = {}

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            img = cv2.imread(img_path)
            embedding = DeepFace.represent(
                img,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append(embedding)
            print(f"✅ Encoded {img_name}")

        except Exception as e:
            print(f"❌ Error {img_name}: {e}")

    if embeddings:
        known_encodings[person_name] = embeddings

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(known_encodings, f)

print("🎉 Face encodings saved successfully")
