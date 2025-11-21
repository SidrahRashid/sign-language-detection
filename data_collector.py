import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime

DATA_PATH = os.path.join('MP_Data') 
ACTIONS = np.array(['hello', 'thanks', 'i_love_you', 'please']) 
NO_SEQUENCES = 75    
SEQUENCE_LENGTH = 30 
KEYPOINT_DIM = 63   

for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Extracts features from the first detected hand, returning zeros otherwise."""
    rh = np.zeros(KEYPOINT_DIM)
    if results.multi_hand_landmarks:
        first_hand_landmarks = results.multi_hand_landmarks[0]
        rh = np.array([[res.x, res.y, res.z] for res in first_hand_landmarks.landmark]).flatten()
    return rh

# --- VIDEO CAPTURE & DATA LOOP ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("FATAL: Camera failed to open. Check index or connection.")
    exit()
    
print(f"Starting data collection for {len(ACTIONS)} words...")

for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret: continue

            # MediaPipe processing
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks for visualization
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detected_status = "DETECTED!"
            else:
                detected_status = "Searching..."

            # Apply Logic
            if frame_num == 0: 
                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(1200) 

            cv2.putText(image, f'Collecting for {action} Video: {sequence}', (15,12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, detected_status, (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break #'q' to break inner loop

cap.release()
cv2.destroyAllWindows()
print("\nData Collection Complete! Proceed to Model Training.")