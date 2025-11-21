import cv2
import mediapipe as mp
import time
from datetime import datetime
import numpy as np 
from tensorflow.keras.models import load_model 

ACTIONS = np.array(['hello', 'thanks', 'i_love_you', 'please']) 
KEYPOINT_DIM = 63 # 21 landmarks * 3 coords

try:
    model = load_model('dynamic_lstm_model.h5')
except Exception as e:
    print(f"FATAL ERROR: Could not load dynamic_lstm_model.h5: {e}")
    print("Ensure 'dynamic_lstm_model.h5' is in the same directory.")
    exit()

SEQUENCE_LENGTH = 30 
sequence = []       
threshold = 0.6  

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

START_HOUR = 18 
END_HOUR = 22

cap = cv2.VideoCapture(0) # '0' typically means the default camera
if not cap.isOpened():
    print("FATAL ERROR: Camera failed to open.")
    exit()
    
print("Starting Sign Language Detector...")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Get current system time
    now = datetime.now()
    current_hour = now.hour
    
    # Check if we are within the operational time window
    is_active = START_HOUR <= current_hour < END_HOUR

    image_display = cv2.flip(image, 1) # Flip the image for selfie view

    if is_active:
        image_processed = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
        image_processed.flags.writeable = False
        
        results = hands.process(image_processed)
        
        image_display.flags.writeable = True
        
        prediction_text = "Buffering frames..." 

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_display, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:] 

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                predicted_action_index = np.argmax(res)
                
                # Check if the confidence exceeds the threshold
                if res[predicted_action_index] > threshold:
                    predicted_word = ACTIONS[predicted_action_index]
                    confidence = res[predicted_action_index]
                    prediction_text = f"PREDICTED: {predicted_word} ({confidence*100:.1f}%)"
                else:
                    prediction_text = "PREDICTED: Unknown"
        
        status_text = f"ACTIVE (6PM-10PM) | {prediction_text}"
        
    else:
        # System is outside of the active window
        status_text = "OFFLINE (Operational: 6PM-10PM)"
        sequence = [] 

    cv2.putText(image_display, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Sign Language Detector', image_display)

    # Exit on 'q' press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detector closed.")