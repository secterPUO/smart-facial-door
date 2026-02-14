from flask import Flask, request, render_template_string, redirect
import cv2
import numpy as np
import face_recognition
import base64
import pickle
import os
import time

app = Flask(__name__)

SAVE_PATH = "enrolled_faces.pkl"

# ================= GLOBAL VARIABLES =================
enrolled_faces = []
last_image_b64 = ""
detect_mode = False
last_result = "READY"
result_timestamp = 0

# ================= LOAD SAVED FACES =================
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, 'rb') as f:
        enrolled_faces = pickle.load(f)
    print(f"Loaded {len(enrolled_faces)} faces")
else:
    print("No saved faces found. Starting fresh.")

# ================= UI =================
HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Smart Facial Door</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body { background:#0f172a; font-family:Arial; color:white; text-align:center; }
        .container { width:80%; margin:auto; margin-top:30px; padding:20px; background:#1e293b; border-radius:15px; }
        .status { padding:15px; border-radius:10px; font-size:20px; margin:15px 0; }
        .authorized { background:#16a34a; }
        .unauthorized { background:#dc2626; }
        .ready { background:#facc15; color:black; }
        button { padding:12px 25px; font-size:16px; border:none; border-radius:8px; margin:10px; cursor:pointer; }
        .btn-detect { background:#3b82f6; color:white; }
        .btn-enroll { background:#9333ea; color:white; }
        img { width:60%; border-radius:12px; border:4px solid #3b82f6; margin-top:15px; }
    </style>
</head>
<body>
<div class="container">
<h1>SMART FACIAL DOOR SYSTEM</h1>

{% if img %}
<img src="data:image/jpeg;base64,{{ img }}">
{% else %}
<p>Waiting for camera...</p>
{% endif %}

{% if result == "AUTHORIZED" %}
<div class="status authorized">ACCESS GRANTED</div>
{% elif result == "UNAUTHORIZED" %}
<div class="status unauthorized">ACCESS DENIED</div>
{% else %}
<div class="status ready">SYSTEM READY</div>
{% endif %}

<p>Enrolled Faces: {{ count }}</p>

<form action="/start_detect" method="post">
<button class="btn-detect">START DETECTION</button>
</form>

<form action="/enroll" method="post">
<button class="btn-enroll">ENROLL FACE</button>
</form>

</div>
</body>
</html>
'''

@app.route("/view")
def view():
    global last_result, result_timestamp

    if last_result in ["AUTHORIZED", "UNAUTHORIZED"]:
        if time.time() - result_timestamp > 3:
            last_result = "READY"

    return render_template_string(
        HTML_PAGE,
        img=last_image_b64,
        result=last_result,
        count=len(enrolled_faces)
    )

@app.route("/start_detect", methods=["POST"])
def start_detect():
    global detect_mode
    detect_mode = True
    return redirect("/view")

@app.route("/enroll", methods=["POST"])
def enroll():
    global last_image_b64

    if not last_image_b64:
        return redirect("/view")

    img_bytes = base64.b64decode(last_image_b64)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb)

    if len(encodings) > 0:
        enrolled_faces.append(encodings[0])
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(enrolled_faces, f)

    return redirect("/view")

@app.route("/recognize", methods=["POST"])
def recognize():
    global last_image_b64, detect_mode, last_result, result_timestamp

    file = request.files["image"].read()
    last_image_b64 = base64.b64encode(file).decode("utf-8")

    if not detect_mode:
        return "IDLE", 200

    if not enrolled_faces:
        detect_mode = False
        return "NO FACES ENROLLED", 400

    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (0,0), fx=0.5, fy=0.5)

    locations = face_recognition.face_locations(small)
    encodings = face_recognition.face_encodings(small, locations)

    detect_mode = False

    if len(encodings) > 0:
        distances = face_recognition.face_distance(enrolled_faces, encodings[0])
        best_match = np.argmin(distances)

        if distances[best_match] < 0.4:
            last_result = "AUTHORIZED"
            result_timestamp = time.time()
            return "MATCH FOUND", 200

    last_result = "UNAUTHORIZED"
    result_timestamp = time.time()
    return "UNKNOWN", 401

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
