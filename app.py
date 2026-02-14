from flask import Flask, request, render_template_string, redirect
import os
import base64
import cv2
import numpy as np
from deepface import DeepFace
import time

app = Flask(__name__)

UPLOAD_FOLDER = "faces"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

last_image_b64 = ""
last_result = "SYSTEM READY"
detect_mode = False


# ========================= HTML TEMPLATE =========================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SMART FACIAL DOOR SYSTEM</title>
<style>
body { background:#0f1c2e; color:white; text-align:center; font-family:Arial; }
.container { margin-top:40px; }
img { width:320px; border-radius:10px; border:3px solid #4da3ff; }
.status { margin:20px; padding:15px; border-radius:10px; font-weight:bold; }
.ready { background:#f1c40f; color:black; }
.granted { background:#2ecc71; }
.denied { background:#e74c3c; }
button { padding:12px 25px; margin:10px; font-size:16px; border:none; border-radius:8px; cursor:pointer; }
.start { background:#3498db; color:white; }
.enroll { background:#8e44ad; color:white; }
</style>
</head>
<body>
<h1>SMART FACIAL DOOR SYSTEM</h1>

<div class="container">
    {% if image %}
        <img src="data:image/jpeg;base64,{{ image }}">
    {% endif %}

    <div class="status {{ status_class }}">
        {{ result }}
    </div>

    <form action="/start" method="post">
        <button class="start">START DETECTION</button>
    </form>

    <form action="/enroll" method="post">
        <button class="enroll">ENROLL FACE</button>
    </form>

    <p>Enrolled Faces: {{ count }}</p>
</div>
</body>
</html>
"""


# ========================= HOME =========================
@app.route("/")
def home():
    status_class = "ready"
    if last_result == "ACCESS GRANTED":
        status_class = "granted"
    elif last_result == "ACCESS DENIED":
        status_class = "denied"

    return render_template_string(
        HTML_PAGE,
        image=last_image_b64,
        result=last_result,
        count=len(os.listdir(UPLOAD_FOLDER)),
        status_class=status_class
    )


# ========================= START DETECTION =========================
@app.route("/start", methods=["POST"])
def start():
    global detect_mode
    detect_mode = True
    return redirect("/")


# ========================= ENROLL =========================
@app.route("/enroll", methods=["POST"])
def enroll():
    global last_result
    last_result = "SHOW YOUR FACE TO CAMERA"

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        filename = f"{UPLOAD_FOLDER}/face_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        last_result = "FACE ENROLLED"

    return redirect("/")


# ========================= DETECT =========================
@app.route("/detect", methods=["POST"])
def detect():
    global last_image_b64, last_result, detect_mode

    if not detect_mode:
        return "IDLE"

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "NO CAMERA"

    # Convert image for web display
    _, buffer = cv2.imencode('.jpg', frame)
    last_image_b64 = base64.b64encode(buffer).decode()

    try:
        result = DeepFace.find(
            img_path=frame,
            db_path=UPLOAD_FOLDER,
            enforce_detection=False
        )

        if len(result[0]) > 0:
            last_result = "ACCESS GRANTED"
            detect_mode = False
            return "UNLOCK"
        else:
            last_result = "ACCESS DENIED"

    except:
        last_result = "ACCESS DENIED"

    return "NO MATCH"


# ========================= RUN =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
