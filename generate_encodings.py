import os
import cv2
import pickle
import numpy as np
from PIL import Image
from deepface import DeepFace

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

        try:
            image = Image.open(img_path)
            rgb = np.array(image.convert('RGB'))
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        except Exception as e:
            print(f"   ❌ Failed to read image: {e}")
            continue

        try:
            image = Image.open(img_path)
            rgb = np.array(image.convert('RGB'))
        except Exception as e:
            print(f"   ❌ Failed to read image: {e}")
            continue

        print(f"   RGB shape: {rgb.shape}, dtype: {rgb.dtype}")

        try:
            # Use DeepFace to detect and encode face
            result = DeepFace.represent(img_path, model_name='VGG-Face', enforce_detection=True)
            if isinstance(result, list) and len(result) > 0:
                encoding = result[0]['embedding']
            else:
                print("   ⚠️ No face detected or encoded")
                continue
        except Exception as e:
            print(f"   ❌ Error processing face: {e}")
            continue

        known_encodings.append(encoding)
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