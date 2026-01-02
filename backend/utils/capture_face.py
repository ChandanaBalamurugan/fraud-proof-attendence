import cv2
import os

# Ask for user name
name = input("Enter your name: ")

# Create folder to save face images
save_path = f"../data/faces/{name}"  # relative to utils folder
os.makedirs(save_path, exist_ok=True)

# Open camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
print("Press Q to stop after capturing faces")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop the **color** face, not gray
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        count += 1

        # Save image
        cv2.imwrite(f"{save_path}/img_{count}.jpg", face)

        # Draw rectangle for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 25:
        break

cap.release()
cv2.destroyAllWindows()
print(f"{count} face images saved for {name}")




