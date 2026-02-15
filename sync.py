import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import sys

print("🚀 Starting Firebase sync script...")

# ------------------ Correct Absolute CSV Path ------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "backend", "data", "attendance.csv")

print("Looking for file at:", CSV_FILE)
print("File exists:", os.path.exists(CSV_FILE))

processed_rows = set()

# ------------------ Firebase Init ------------------ #
try:
    cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase Connected Successfully")
except Exception as e:
    print("❌ Firebase Error:", e)
    sys.exit()

# ------------------ Upload Existing Rows at Startup ------------------ #
def upload_existing_rows():
    """Upload any existing CSV rows that aren't in Firebase yet."""
    global processed_rows
    if not os.path.exists(CSV_FILE):
        print("❌ CSV file missing.")
        return

    try:
        df = pd.read_csv(CSV_FILE, dtype=str).fillna('')
        print(f"📊 Total rows in CSV at startup: {len(df)}")
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            date = str(row.get("Date", "")).strip()
            time_value = str(row.get("Time", "")).strip()
            if not name or not date or not time_value:
                continue  # skip incomplete rows

            doc_id = f"{name}_{date}_{time_value}"
            if doc_id not in processed_rows:
                # Upload to Firebase
                success = False
                retries = 3
                while not success and retries > 0:
                    try:
                        db.collection("attendance").document(doc_id).set({
                            "Name": name,
                            "Date": date,
                            "Time": time_value,
                            "Timestamp": firestore.SERVER_TIMESTAMP
                        })
                        processed_rows.add(doc_id)
                        print(f"✅ Uploaded existing row: {doc_id}")
                        success = True
                    except Exception as e:
                        retries -= 1
                        print(f"⚠️ Retry failed for {doc_id}, retries left {retries}: {e}")
                        time.sleep(1)
                if not success:
                    print(f"❌ Could not upload existing row after retries: {doc_id}")
        print(f"📊 Loaded and processed {len(processed_rows)} rows at startup.")
    except Exception as e:
        print("❌ Error reading CSV at startup:", e)

upload_existing_rows()

# ------------------ Watch for New Rows ------------------ #
def check_for_new_rows():
    global processed_rows

    if not os.path.exists(CSV_FILE):
        print("❌ CSV file missing.")
        return

    try:
        df = pd.read_csv(CSV_FILE, dtype=str).fillna('')
    except Exception as e:
        print("❌ Error reading CSV:", e)
        return

    print("\n📂 CSV modified. Checking for new rows...")
    print("Total rows in CSV:", len(df))
    print("Already processed rows:", len(processed_rows))

    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        date = str(row.get("Date", "")).strip()
        time_value = str(row.get("Time", "")).strip()
        if not name or not date or not time_value:
            continue

        doc_id = f"{name}_{date}_{time_value}"
        print("Checking:", doc_id)

        if doc_id not in processed_rows:
            print("🔥 New row detected!")
            success = False
            retries = 3
            while not success and retries > 0:
                try:
                    db.collection("attendance").document(doc_id).set({
                        "Name": name,
                        "Date": date,
                        "Time": time_value,
                        "Timestamp": firestore.SERVER_TIMESTAMP
                    })
                    processed_rows.add(doc_id)
                    print(f"✅ NEW ROW ADDED: {doc_id}")
                    success = True
                except Exception as e:
                    retries -= 1
                    print(f"⚠️ Firebase upload failed for {doc_id}, retries left {retries}: {e}")
                    time.sleep(1)
            if not success:
                print(f"❌ Could not upload row after retries: {doc_id}")

# ------------------ Watchdog Handler ------------------ #
class CSVHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("attendance.csv"):
            time.sleep(1)  # Small delay to avoid double triggers
            check_for_new_rows()

observer = Observer()
observer.schedule(
    CSVHandler(),
    path=os.path.join(BASE_DIR, "backend", "data"),
    recursive=False
)
observer.start()

# ------------------ Keep Script Running ------------------ #
print("👀 Watching for NEW rows only... Press CTRL+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Stopping script...")
    observer.stop()
observer.join()
print("🛑 Firebase sync stopped.")

