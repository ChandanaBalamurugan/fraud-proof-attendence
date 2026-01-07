import dlib
print("dlib version:", dlib.__version__)
detector = dlib.get_frontal_face_detector()
import numpy as np
# Create a dummy image
img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
try:
    dets = detector(img)
    print("Dets:", dets)
except Exception as e:
    print("Error:", e)