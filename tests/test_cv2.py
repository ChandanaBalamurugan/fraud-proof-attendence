import cv2
print("OpenCV version:", cv2.__version__)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
else:
    ret, frame = cap.read()
    if ret:
        print("Camera works! Frame size:", frame.shape)
    else:
        print("Failed to read frame")
cap.release()
