import cv2
import time

cap = cv2.VideoCapture(0)
time.sleep(1) # Give camera time to initialize

if not cap.isOpened():
    print("Test FAILED: Camera could not be opened by OpenCV.")
else:
    ret, frame = cap.read()
    if ret:
        print(f"Test SUCCESS: Frame captured with shape: {frame.shape}")
    else:
        print("Test FAILED: Camera opened but failed to read a frame.")

cap.release()