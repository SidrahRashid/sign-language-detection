<<<<<<< HEAD
# 🤟 Real-Time Sign Language Detection System
> Flask • MediaPipe • LSTM — Real-time ASL recognition with an elegant UI and image upload
=======
**🤟 Real-Time Sign Language Detection System (Flask + MediaPipe + LSTM)**
>>>>>>> a8baf785b3145b7212fc4d6f8c10786412d63986

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Flask](https://img.shields.io/badge/flask-2.0-lightgrey.svg)]()
[![TensorFlow](https://img.shields.io/badge/tensorflow-keras-orange.svg)]()
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)]()

---

<<<<<<< HEAD
## ✨ Project Overview
=======
**🚀 Features**
🎥 1. Real-Time Video Sign Detection
>>>>>>> a8baf785b3145b7212fc4d6f8c10786412d63986

**Real-Time Sign Language Detection System** is an end-to-end project that recognizes a handful of American Sign Language gestures using MediaPipe hand landmarks and an LSTM model. The system supports both **real-time webcam detection** and **single-image upload**, with a polished Flask-based GUI and time-based activation.

Key features:
- Live webcam feed with landmark overlays and smooth predictions.
- Image upload for single-frame analysis.
- Threaded architecture for non-blocking prediction and streaming.
- Time-window control for availability (e.g., 6 PM – 10 PM).
- Clean, modern UI with glassmorphism + soft design.

---

## 🎯 What it Demonstrates

- Real-time computer vision (MediaPipe Hands)
- Sequential deep learning (LSTM with Keras/TensorFlow)
- Production-style engineering (Flask, threading, MJPEG streaming)
- UX/UI for ML applications
- Debugging and robustness (stale buffer handling, smoothing)

---

## 🧭 Quick Links

- **Local Flask app code**: `/mnt/data/app.py`  
  (open this file to review or tweak the backend quickly)

---

## 📁 Repository Structure

project_root/
├─ app.py # Flask web app + streaming + prediction
├─ dynamic_lstm_model.h5 # Trained LSTM model (not tracked in Git)
├─ mp_data/ # (excluded via .gitignore) your dataset images
├─ requirements.txt # pip dependencies
├─ README.md # (this file)
└─ static/ # optional css/js assets

yaml
Copy code

---

## 🚀 Demo / Run Locally

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / Mac
   venv\Scripts\activate      # Windows
Install dependencies:

<<<<<<< HEAD
bash
Copy code
=======
Prediction worker thread: Runs ML inference

Video generator: Handles camera + landmark drawing

No blocking, no lag

🧼 5. Smart Input Buffering

Automatically clears buffer when:

No hand detected for long

Hand landmarks vanish

Stale frames detected

Guarantees high accuracy

**🧠 Tech Stack**
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

**📝 How It Works**
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
>>>>>>> a8baf785b3145b7212fc4d6f8c10786412d63986
pip install -r requirements.txt
Start the app:

bash
Copy code
python app.py
Open the UI:

<<<<<<< HEAD
cpp
Copy code
http://127.0.0.1:5000
=======
Step 3: Open in browser
http://127.0.0.1:5000
>>>>>>> a8baf785b3145b7212fc4d6f8c10786412d63986
