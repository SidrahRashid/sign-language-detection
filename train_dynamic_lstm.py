import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GaussianNoise, TimeDistributed
from tensorflow.keras.optimizers import Adam

SEQUENCE_DIR = 'MP_Data' 
ACTIONS = np.array(['hello', 'thanks', 'i_love_you', 'please']) 
SEQUENCE_LENGTH = 30
NO_SEQUENCES = 50     
KEYPOINT_DIM = 63

label_map = {label: num for num, label in enumerate(ACTIONS)}
NUM_CLASSES = len(ACTIONS)
print(f"Found {NUM_CLASSES} classes for LSTM training.")

sequences, labels = [], []
print("Loading sequence data from disk...")

for action in ACTIONS:
    action_dir = os.path.join(SEQUENCE_DIR, action)
    if not os.path.exists(action_dir): continue 
        
    for sequence in range(NO_SEQUENCES): 
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            npy_path = os.path.join(action_dir, str(sequence), str(frame_num) + '.npy')
            try:
                window.append(np.load(npy_path))
            except:
                continue # Skip if file is somehow missing

        if window:
            sequence_array = np.array(window)
            if np.any(sequence_array != 0):
                sequences.append(sequence_array)
                labels.append(label_map[action])

if not sequences:
    print("FATAL ERROR: No valid, non-zero sequences loaded. Re-run data collection.")
    exit()

X = np.array(sequences) 
y = to_categorical(labels).astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
print(f"Total valid sequences loaded: {len(X)}. Training on {X_train.shape[0]} sequences.")

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, KEYPOINT_DIM)))
model.add(GaussianNoise(0.005, input_shape=(SEQUENCE_LENGTH, KEYPOINT_DIM)))
model.add(TimeDistributed(Dense(32, activation='relu'), 
          input_shape=(SEQUENCE_LENGTH, KEYPOINT_DIM))) # NEW LAYER
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax')) 

opt = Adam(learning_rate=0.001) 
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()

print("Starting LSTM training...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

model_save_path = 'dynamic_lstm_model.h5'
model.save(model_save_path)
print(f"Dynamic LSTM model saved to {model_save_path}")

res = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Evaluation (Test Set):")
print(f"  Accuracy: {res[1]*100:.2f}%")