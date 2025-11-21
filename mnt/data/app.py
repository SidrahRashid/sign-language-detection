# /mnt/data/app.py
from flask import Flask, Response, render_template_string, request
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import threading
import time
import atexit
import os

# CONFIG 
ACTIONS = np.array(['hello', 'thanks', 'i_love_you', 'please'])
SEQUENCE_LENGTH = 30
KEYPOINT_DIM = 63
START_HOUR = 18
END_HOUR = 22
THRESHOLD = 0.6

app = Flask(__name__)
DEBUG_MODE = False

sequence_lock = threading.Lock()
prediction_lock = threading.Lock()

sequence_buffer = []
prediction_text = "Detector Initialized"

last_valid_kp = np.zeros(KEYPOINT_DIM, dtype=np.float32)
last_hand_ts = 0.0
NO_HAND_CLEAR_SECONDS = 1.5
buffer_was_cleared = False  

# smoothing
PROB_HISTORY = []
PROB_HISTORY_SIZE = 5

MODEL = None
MODEL_LOADED = False
MODEL_PATH = "dynamic_lstm_model.h5"

mp_hands = mp.solutions.hands
hands = None
mp_drawing = None

try:
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH)
        MODEL_LOADED = True
        try:
            MODEL.predict(np.zeros((1, SEQUENCE_LENGTH, KEYPOINT_DIM), dtype=np.float32), verbose=0)
        except Exception:
            pass
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}")
except Exception as e:
    print("[ERROR] loading model:", e)
    MODEL_LOADED = False

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def cleanup_camera(cap):
    try:
        if cap is not None and cap.isOpened():
            cap.release()
            if DEBUG_MODE:
                print("[CLEANUP] camera released")
    except Exception as e:
        print("[CLEANUP] error:", e)

atexit.register(lambda: None)

def now_is_active():
    return START_HOUR <= datetime.now().hour < END_HOUR

def extract_keypoints_from_results(results):
    """
    Return RAW flattened (63,) keypoints exactly like data_collector.py:
    x0,y0,z0, x1,y1,z1, ... (21*3)
    """
    keypoints = np.zeros(KEYPOINT_DIM, dtype=np.float32)
    if results and results.multi_hand_landmarks:
        first = results.multi_hand_landmarks[0]
        pts = []
        for lm in first.landmark:
            pts.append([lm.x, lm.y, lm.z])
        keypoints = np.array(pts, dtype=np.float32).flatten()
    return keypoints, []

def model_predict_probs_from_sequence(seq_list):
    if not MODEL_LOADED:
        return None
    seq_arr = np.array(seq_list, dtype=np.float32)
    seq_arr = np.expand_dims(seq_arr, axis=0)
    probs = MODEL.predict(seq_arr, verbose=0)[0]
    return probs

def prediction_worker():
    global sequence_buffer, prediction_text, last_hand_ts, PROB_HISTORY, buffer_was_cleared, last_valid_kp
    print("[WORKER] started")
    while True:
        time.sleep(0.06)
        if not now_is_active():
            with prediction_lock:
                prediction_text = "SYSTEM OFFLINE"
            continue
        if not MODEL_LOADED:
            with prediction_lock:
                prediction_text = "MODEL NOT LOADED"
            time.sleep(1.0)
            continue

        if (time.time() - last_hand_ts) > NO_HAND_CLEAR_SECONDS:
            with sequence_lock:
                if sequence_buffer:
                    if DEBUG_MODE:
                        print("[DBG] clearing buffer due to no-hand timeout")
                    sequence_buffer.clear()
            PROB_HISTORY = []
            buffer_was_cleared = True
            last_valid_kp = np.zeros(KEYPOINT_DIM, dtype=np.float32)
            with prediction_lock:
                prediction_text = "Searching for hand..."
            time.sleep(0.08)
            continue

        with sequence_lock:
            buf_len = len(sequence_buffer)
            if buf_len < SEQUENCE_LENGTH:
                continue
            seq_copy = sequence_buffer[-SEQUENCE_LENGTH:].copy()

        last_kp = seq_copy[-1]
        if not np.any(last_kp) or np.mean(np.abs(last_kp)) < 1e-5:
            if DEBUG_MODE:
                print("[DBG] skipping prediction because last_kp is essentially zero")
            with prediction_lock:
                prediction_text = "Searching for hand..."
            time.sleep(0.08)
            continue

        try:
            probs = model_predict_probs_from_sequence(seq_copy)
            if probs is None:
                with prediction_lock:
                    prediction_text = "MODEL NOT LOADED"
                time.sleep(0.2)
                continue

            PROB_HISTORY.append(probs)
            if len(PROB_HISTORY) > PROB_HISTORY_SIZE:
                PROB_HISTORY.pop(0)
            avg_probs = np.mean(PROB_HISTORY, axis=0)
            top_idx = int(np.argmax(avg_probs))
            top_prob = float(avg_probs[top_idx])

            if DEBUG_MODE:
                try:
                    print("[DBG] avg_probs =", ["{:.4f}".format(x) for x in avg_probs.tolist()])
                except Exception:
                    pass

            if top_prob >= THRESHOLD:
                predicted_word = ACTIONS[top_idx]
                confidence = top_prob
                with prediction_lock:
                    prediction_text = f"PREDICTED: {predicted_word} ({confidence*100:.1f}%)"
                if DEBUG_MODE:
                    print(f"[DBG] prediction accepted: {predicted_word} ({confidence:.4f})")
            else:
                with prediction_lock:
                    prediction_text = "PREDICTED: Unknown"
                if DEBUG_MODE:
                    print(f"[DBG] prediction rejected top_prob={top_prob:.4f} < threshold={THRESHOLD:.4f}")

            time.sleep(0.12)
        except Exception as e:
            print("[WORKER] prediction error:", e)
            with prediction_lock:
                prediction_text = "ERROR: predict failed"
            time.sleep(0.5)

def generate_frames():
    global sequence_buffer, prediction_text, last_hand_ts, last_valid_kp, buffer_was_cleared
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[FATAL] Camera could not be opened.")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                if DEBUG_MODE:
                    print("[GENERATOR] frame read failed, breaking")
                break

            frame_flipped = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            frame_display = frame_flipped.copy()

            if results and results.multi_hand_landmarks:
                buffer_was_cleared = False
                last_hand_ts = time.time()
                kp, _ = extract_keypoints_from_results(results)
                if np.any(kp):
                    last_valid_kp = kp.copy()
                with sequence_lock:
                    sequence_buffer.append(kp)
                    if len(sequence_buffer) > SEQUENCE_LENGTH * 3:
                        sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]
                    if DEBUG_MODE:
                        print("[DBG] appended kp; new buffer length:", len(sequence_buffer))
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0,128,255), thickness=2)
                    )
            else:
                with sequence_lock:
                    if not buffer_was_cleared:
                        sequence_buffer.append(last_valid_kp.copy())
                        if DEBUG_MODE:
                            print("[DBG] appended last_valid_kp; new buffer length:", len(sequence_buffer))
                    else:
                        sequence_buffer.append(np.zeros(KEYPOINT_DIM, dtype=np.float32))
                        if DEBUG_MODE:
                            print("[DBG] appended zero (buffer_was_cleared); new buffer length:", len(sequence_buffer))
                    if len(sequence_buffer) > SEQUENCE_LENGTH * 3:
                        sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]

            with prediction_lock:
                text = prediction_text

            cv2.rectangle(frame_display, (5, 5), (640, 40), (0, 0, 0), -1)
            cv2.putText(frame_display, text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame_display)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cleanup_camera(camera)

@app.route('/')
def index():
    is_active = now_is_active()
    status = "ONLINE" if is_active else "OFFLINE"
    status_color = "green" if is_active else "red"
    return render_template_string('''
<!-- Paste this string into your render_template_string(...) in app.py -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>ASL Detector — Dashboard</title>

  <!-- Google Font -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">

  <style>
    :root{
      --bg: #f6f7fb;
      --card: rgba(255,255,255,0.65);
      --glass-border: rgba(255,255,255,0.6);
      --muted: #7b7f88;
      --accent: #009975; /* soft green */
      --accent-2: #6dd3b5;
      --shadow: 0 8px 24px rgba(16,24,40,0.08);
      --glass-blur: 10px;
      --radius: 14px;
    }
    html,body{height:100%;margin:0;font-family:'Poppins',system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:linear-gradient(180deg,#f8fafc 0%, #eef3f8 100%);color:#111;}
    .wrap{max-width:1100px;margin:28px auto;padding:26px;}
    header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
    .brand{display:flex;gap:12px;align-items:center}
    .logo{
      width:54px;height:54px;border-radius:12px;
      background:linear-gradient(135deg,var(--accent),var(--accent-2));
      box-shadow: 0 6px 18px rgba(3,80,62,0.12);
      display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:22px;
    }
    h1{margin:0;font-size:20px;}
    .subtitle{font-size:13px;color:var(--muted);margin-top:4px}

    /* Status pill */
    .controls{display:flex;gap:12px;align-items:center}
    .pill{
      padding:8px 12px;border-radius:999px;font-weight:600;font-size:13px;
      display:inline-flex;align-items:center;gap:8px;border:1px solid rgba(0,0,0,0.04);
    }
    .pill.online{background:linear-gradient(90deg,#e8ffef,#f0fff6);color: #05643b;border-color: rgba(5,100,59,0.08)}
    .pill.offline{background:#fff5f5;color:#8b1d1d;border-color: rgba(139,29,29,0.06)}

    /* layout */
    .grid{display:grid;grid-template-columns: 1fr 360px;gap:20px}
    @media (max-width:920px){ .grid{grid-template-columns: 1fr;}}
    .card{
      background:var(--card);
      border-radius:var(--radius);
      box-shadow:var(--shadow);
      padding:16px;
      backdrop-filter: blur(var(--glass-blur));
      border: 1px solid var(--glass-border);
    }

    /* Video box */
    .video-wrap{display:flex;flex-direction:column;gap:12px}
    .video-frame{width:100%;height: auto;border-radius:10px;overflow:hidden;border:1px solid rgba(12,17,23,0.06); background:#000;display:flex;align-items:center;justify-content:center}
    .video-frame img{width:100%;height:auto;display:block}

    .prediction{
      display:flex;align-items:center;justify-content:space-between;padding:10px;border-radius:10px;background:linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0.45));border:1px solid rgba(0,0,0,0.03);
      font-weight:600;margin-top:6px;
    }

    /* Right column */
    .side {display:flex;flex-direction:column;gap:16px}
    .upload-area{display:flex;flex-direction:column;gap:8px}
    .upload-btn{
      display:inline-block;padding:10px 14px;border-radius:10px;border:1px dashed rgba(0,0,0,0.08);
      background:linear-gradient(180deg,#ffffff,#fbfcff);
      cursor:pointer;color:#334155;font-weight:600;
    }

    .meta{font-size:13px;color:var(--muted)}
    .small{font-size:12px;color:var(--muted)}

    /* footer */
    .footer{margin-top:18px;font-size:13px;color:var(--muted);display:flex;justify-content:space-between;align-items:center}
    .link{color:#0b6b53;font-weight:600;text-decoration:none}

    /* nice subtle hover */
    .upload-btn:hover{box-shadow: 0 6px 18px rgba(3,80,62,0.06)}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="brand">
        <div class="logo">AS</div>
        <div>
          <h1>Sign Language Detector</h1>
          <div class="subtitle">Real-time detection • Image upload • Time-limited operation</div>
        </div>
      </div>

      <div class="controls">
        <div class="pill {{ 'online' if status == 'ONLINE' else 'offline' }}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="opacity:0.9">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="0.5" fill="currentColor" style="opacity:0.15"></circle>
            <path d="M5 12a7 7 0 0 1 14 0" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          SYSTEM STATUS: <span style="margin-left:6px;color:inherit">{{ status }}</span>
        </div>
      </div>
    </header>

    <main class="grid">
      <!-- Left: Video -->
      <section class="card video-wrap">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <div style="font-weight:700">Live Camera</div>
          <div class="small">Active window: {{ 0 if status=='OFFLINE' else START_HOUR }} — {{ END_HOUR }} </div>
        </div>

        <div class="video-frame" aria-hidden="true">
          <!-- video_feed route (MJPEG) -->
          <img src="{{ url_for('video_feed') }}" alt="Live video feed">
        </div>

        <div class="prediction">
          <div>Current prediction:</div>
          <div id="pred-text" style="color:var(--accent)">{{ "Loading..." }}</div>
        </div>

        <div style="display:flex;gap:10px;margin-top:8px;align-items:center">
          <div class="small">Tip: keep your hand centered and 6–12 inches from camera.</div>
        </div>
      </section>

      <!-- Right: Upload and Info -->
      <aside class="side">
        <div class="card upload-area">
          <div style="font-weight:700">Upload Image</div>
          <div class="meta">Analyze a single frame (LSTM padded). Use JPG/PNG.</div>

          <form action="/upload" method="post" enctype="multipart/form-data" style="display:flex;flex-direction:column;gap:8px">
            <input type="file" name="file" accept=".jpg,.png,.jpeg" style="padding:8px;border-radius:8px;border:1px solid rgba(0,0,0,0.06);background:white">
            <button type="submit" class="upload-btn">Analyze Image</button>
          </form>

          <div class="small">Last result: <span id="upload-result">—</span></div>
        </div>

        <div class="card">
          <div style="font-weight:700;margin-bottom:8px">About</div>
          <div class="small">
            This interface uses MediaPipe for keypoint extraction and an LSTM model for temporal prediction.
            The app is active only during the configured time window.
          </div>

          <div style="margin-top:12px;display:flex;gap:8px">
            <a class="link" href="/mnt/data/app.py" download>Download app.py</a>
            <a class="link" href="#" onclick="alert('Demo only — attach help docs here')">Help</a>
          </div>
        </div>

        <div class="card" style="text-align:center;font-size:13px;color:var(--muted)">
          <div style="font-weight:700">Model</div>
          <div class="small" style="margin-top:8px">dynamic_lstm_model.h5</div>
          <div style="margin-top:6px;font-size:12px">Threshold: {{ THRESHOLD }}</div>
        </div>
      </aside>
    </main>

    <div class="footer">
      <div>Built with ❤️ — keep testing in different lighting</div>
      <div class="small">Need UI tweaks? Say “tweak colors” or “compact layout”</div>
    </div>
  </div>

  <script>
    // Simple frontend poll to fetch prediction text from the rendered video overlay
    // (Non-invasive — uses the text snapshot already written by server into video overlay)
    // If you want real-time text via socket, we can add socket.io later.
    (function pollPrediction(){
      try{
        const predEl = document.getElementById('pred-text');
        // read the image element (the overlay is baked into MJPEG image) — so we keep static
        // Instead, update the upload-result area by listening to form submit results (simple UX)
      }catch(e){/*ignore*/}
      setTimeout(pollPrediction, 2000);
    })();
  </script>
</body>
</html>

    ''', status=status, status_color=status_color)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    data = file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "Could not decode image", 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    kp, _ = extract_keypoints_from_results(res)
    if not np.any(kp):
        return "No hands detected", 200

    seq = np.zeros((SEQUENCE_LENGTH, KEYPOINT_DIM), dtype=np.float32)
    seq[-1, :] = kp

    if not MODEL_LOADED:
        return "Model not loaded", 500

    probs = model_predict_probs_from_sequence(seq)
    if probs is None:
        return "Model error", 500
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    if top_prob < THRESHOLD:
        return f"Unknown ({top_prob:.3f})", 200
    else:
        return f"{ACTIONS[top_idx]} ({top_prob*100:.1f}%)", 200

if __name__ == '__main__':
    worker = threading.Thread(target=prediction_worker, daemon=True)
    worker.start()
    print("Worker started. Launching Flask app on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
