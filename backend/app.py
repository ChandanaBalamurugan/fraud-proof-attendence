from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Firebase initialization
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Simple face recognition using template matching
known_faces = {}
face_names = []

def load_known_faces():
    faces_dir = "data/faces"
    if not os.path.exists(faces_dir):
        print("⚠️ Faces directory not found")
        return

    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if os.path.isdir(person_dir):
            face_images = []
            for img_file in os.listdir(person_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        face_images.append(gray)

            if face_images:
                known_faces[person_name] = face_images
                face_names.append(person_name)
                print(f"✅ Loaded {len(face_images)} images for {person_name}")

load_known_faces()

def recognize_face(face_img):
    """Simple face recognition using template matching"""
    best_match = "Unknown"
    best_score = 0.0

    for name, face_list in known_faces.items():
        for known_face in face_list:
            # Resize faces to same size for comparison
            face_resized = cv2.resize(face_img, (100, 100))
            known_resized = cv2.resize(known_face, (100, 100))

            # Use template matching
            result = cv2.matchTemplate(face_resized, known_resized, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)

            if score > best_score and score > 0.6:  # Threshold for match
                best_score = score
                best_match = name

    return best_match if best_score > 0.6 else "Unknown"

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    try:
        attendance_ref = db.collection('attendance')
        docs = attendance_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).get()

        attendance_data = []
        for doc in docs:
            data = doc.to_dict()
            attendance_data.append({
                'id': doc.id,
                'name': data.get('name', 'Unknown'),
                'timestamp': data.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if data.get('timestamp') else '',
                'method': data.get('method', 'unknown')
            })

        return jsonify({'success': True, 'attendance': attendance_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/mark-attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.json
        name = data.get('name')
        method = data.get('method', 'manual')

        if not name:
            return jsonify({'success': False, 'error': 'Name is required'})

        attendance_data = {
            'name': name,
            'timestamp': datetime.now(),
            'method': method
        }

        db.collection('attendance').add(attendance_data)

        return jsonify({'success': True, 'message': f'Attendance marked for {name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})

        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy array
        frame = np.array(image)

        # Convert RGB to BGR for OpenCV if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if frame.dtype == np.uint8:  # RGB image
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        recognized_names = []

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]

            # Recognize face
            name = recognize_face(face_roi)

            if name != "Unknown" and name not in recognized_names:
                recognized_names.append(name)

        # Mark attendance for recognized faces
        marked_attendance = []
        for name in recognized_names:
            try:
                attendance_data = {
                    'name': name,
                    'timestamp': datetime.now(),
                    'method': 'camera'
                }
                db.collection('attendance').add(attendance_data)
                marked_attendance.append(name)
            except Exception as e:
                print(f"Error marking attendance for {name}: {e}")

        return jsonify({
            'success': True,
            'recognized': recognized_names,
            'marked': marked_attendance,
            'faces_detected': len(faces),
            'message': f'Detected {len(faces)} faces, recognized {len(recognized_names)} people, marked attendance for {len(marked_attendance)}'
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Starting Flask server...")
    print(f"📊 Loaded {len(face_names)} known faces: {face_names}")
    app.run(debug=False, host='0.0.0.0', port=port)
