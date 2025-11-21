🤟 Real-Time Sign Language Detection System (Flask + MediaPipe + LSTM)

A complete end-to-end Sign Language Recognition System built using MediaPipe, TensorFlow (LSTM), and a Flask web interface.
It supports real-time webcam prediction, image upload prediction, and a smart time-based availability window.

🔥 This project demonstrates skills in deep learning, computer vision, threading, real-time systems, backend development, and UI engineering. Perfect for ML/AI portfolios and resumes.

🚀 Features
🎥 1. Real-Time Video Sign Detection

Uses your webcam feed

Runs MediaPipe hand tracking

Draws landmarks directly on the video

Applies LSTM prediction on extracted sequences

Smooth & accurate predictions

Fully optimized to avoid lag

🖼️ 2. Image Upload Prediction

Upload a static image (jpg/png)

MediaPipe extracts keypoints

LSTM performs inference on padded sequences

Instant result shown on UI

⏰ 3. Time-Controlled System Availability

System only works within selected hours
(Example: 6 PM – 10 PM)

Outside this window → “System Offline”

⚙️ 4. Threaded Architecture

Prediction worker thread: Runs ML inference

Video generator: Handles camera + landmark drawing

No blocking, no lag

🧼 5. Smart Input Buffering

Automatically clears buffer when:

No hand detected for long

Hand landmarks vanish

Stale frames detected

Guarantees high accuracy

🧠 Tech Stack
Component	Technology
Hand Tracking	MediaPipe Hands
Sequence Model	LSTM (TensorFlow / Keras)
Backend	Flask
Frontend	Custom HTML Template
Real-Time Video	MJPEG Streaming
Threading	Python threading module
📦 Project Structure
📁 project_root/
│── app.py                 # Flask app with threaded ML pipeline
│── dynamic_lstm_model.h5  # Trained LSTM model
│── mp_data/               # (Excluded from Git using .gitignore)
│── static/                # Optional CSS/JS assets
│── README.md              # Project documentation
│── requirements.txt

📝 How It Works
1️⃣ MediaPipe extracts 21 hand landmark coordinates

→ Each frame gives (21 × 3) = 63 values.

2️⃣ Frames are collected into sequences

→ Buffer size = 30 frames.

3️⃣ LSTM Model predicts one of the actions:

hello
thanks
i_love_you
please

4️⃣ Real-time predictions displayed on top of video
▶️ Run the App
Step 1: Install Dependencies
pip install -r requirements.txt

Step 2: Start the Flask App
python app.py

Step 3: Open in browser
http://127.0.0.1:5000