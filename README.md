# 🤟 Real-Time Sign Language Detection System
> Flask • MediaPipe • LSTM — Real-time ASL recognition with an elegant UI and image upload

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Flask](https://img.shields.io/badge/flask-2.0-lightgrey.svg)]()
[![TensorFlow](https://img.shields.io/badge/tensorflow-keras-orange.svg)]()
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)]()

---

## ✨ Project Overview

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

bash
Copy code
pip install -r requirements.txt
Start the app:

bash
Copy code
python app.py
Open the UI:

cpp
Copy code
http://127.0.0.1:5000