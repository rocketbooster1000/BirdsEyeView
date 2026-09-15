import cv2

for i in range(5):
    print(f"Checking camera index {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Working camera at index {i}")
        cap.release()