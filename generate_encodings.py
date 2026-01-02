import os
import cv2
import face_recognition
import pickle
import numpy as np

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACES_DIR = os.path.join(BASE_DIR, "backend", "data", "faces")
OUTPUT_FILE = os.path.join(BASE_DIR, "backend", "data", "encodings.pkl")

print("======================================")
print("🔍 Face Encoding Started")
print("📂 Faces directory:", FACES_DIR)
print("======================================")

if not os.path.exists(FACES_DIR):
    print("❌ ERROR: faces folder not found")
    exit()

known_encodings = []
known_names = []

# ================= PROCESS =================
for person_name in os.listdir(FACES_DIR):
    person_path = os.path.join(FACES_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"\n👤 Person: {person_name}")

    for img_name in os.listdir(person_path):

        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, img_name)
        print(f"   📸 Reading: {img_name}")

        image = cv2.imread(img_path)

        if image is None:
            print("   ❌ Failed to read image")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        face_locations = face_recognition.face_locations(rgb)

        if len(face_locations) == 0:
            print("   ⚠️ No face detected")
            continue

        encodings = face_recognition.face_encodings(rgb, face_locations)

        if len(encodings) == 0:
            print("   ⚠️ Encoding failed")
            continue

        known_encodings.append(encodings[0])
        known_names.append(person_name)

        print("   ✅ Face encoded")

# ================= SAVE =================
if len(known_encodings) == 0:
    print("\n❌ No faces encoded. Check images.")
    exit()

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("\n======================================")
print("🎉 SUCCESS!")
print("✅ Total faces encoded:", len(known_encodings))
print("📁 Saved file:", OUTPUT_FILE)
print("======================================")