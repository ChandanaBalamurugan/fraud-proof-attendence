import pickle
import numpy as np
import os

# Path to your encodings.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENC_FILE = os.path.join(BASE_DIR, "backend", "data", "encodings.pkl")

print(f"📂 Loading encodings from: {ENC_FILE}")

# Load the pickle
with open(ENC_FILE, "rb") as f:
    known_encodings = pickle.load(f)

# Check types
for person, enc_list in known_encodings.items():
    print(f"Person: {person}")
    for i, enc in enumerate(enc_list):
        print(f"  Face {i} dtype: {enc.dtype}, shape: {enc.shape}")


