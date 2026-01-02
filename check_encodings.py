print("CHECK SCRIPT STARTED")

import pickle

print("Trying to open encodings.pkl...")

with open("backend/data/encodings.pkl", "rb") as f:
    data = pickle.load(f)

print("Loaded data successfully")
print("Type:", type(data))
print("Keys:", data.keys())

