import cv2

# Open webcam (0 is default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("📷 Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Can't receive frame. Exiting...")
        break

    cv2.imshow('Webcam Test', frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quitting webcam test")
        break

cap.release()
cv2.destroyAllWindows()

