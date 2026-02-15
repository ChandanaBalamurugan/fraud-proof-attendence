import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase with the local JSON file
cred = credentials.Certificate("firebase_service_account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Read the CSV file
df = pd.read_csv("data.csv")
records = df.to_dict(orient="records")

# Upload each row as a document
for row in records:
    db.collection("attendance").add(row)

print(f"Uploaded {len(records)} records to Firestore collection 'attendance'")