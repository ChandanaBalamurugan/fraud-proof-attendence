import os
import cv2
import pickle
import numpy as np
import face_recognition

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
faces_dir = os.path.join(BASE_DIR, "data", "faces")
output_file = os.path.join(BASE_DIR, "data", "encodings.pkl")

print(f"📂 Reading faces from: {faces_dir}")

known_encodings = {}
total_faces = 0

# Iterate over people
for person_name in os.listdir(faces_dir):
    person_path = os.path.join(faces_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"👤 Person: {person_name}")
    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Cannot read image: {img_name}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) == 0:
            print(f"⚠️ Encoding failed: {img_name}")
            continue

        # Convert encoding to float64 numpy array
        encoding = np.array(encodings[0], dtype=np.float64)

        if person_name not in known_encodings:
            known_encodings[person_name] = []
        known_encodings[person_name].append(encoding)
        total_faces += 1
        print(f"✅ Encoded: {img_name}")

# Save to pickle
with open(output_file, "wb") as f:
    pickle.dump(known_encodings, f)

print(f"🎉 ENCODING COMPLETE | Total faces: {total_faces} | Saved at {output_file}")







