from flask import Flask, render_template, request, jsonify, Response
import yt_dlp
import cv2
import av
import torch
import threading
import time
from collections import deque
from ultralytics import YOLO
import os

app = Flask(__name__)

# Global Variables
frame_deque = deque(maxlen=10)
frame_lock = threading.Lock()
video_path = ""
capture_thread = None
processing = False

# Load YOLO model
model = YOLO('yolov8n.pt')
model.fuse()
if torch.cuda.is_available():
    model.cuda()

def download_video(youtube_url):
    """Download video from YouTube."""
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': 'video.mp4',
        'noplaylist': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return "video.mp4"
    except Exception as e:
        print(f"Download error: {e}")
        return None

def capture_frames():
    """Capture frames from the downloaded video."""
    global processing
    cap = cv2.VideoCapture(video_path)
    while processing:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        with frame_lock:
            frame_deque.append(frame)
        time.sleep(0.03)
    cap.release()

def process_stream():
    """Process the video frames and apply object detection."""
    while processing:
        img = None
        with frame_lock:
            if frame_deque:
                img = frame_deque.pop()
        if img is not None:
            try:
                results = model(img)
                for result in results:
                    for box in result.boxes.xywh.cpu().numpy():
                        x, y, w, h = map(float, box)
                        cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
                _, buffer = cv2.imencode('.jpg', img)
                return buffer.tobytes()
            except Exception as e:
                print(f"Processing error: {str(e)}")
    return None

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/start', methods=['POST'])
def start_processing():
    global video_path, capture_thread, processing
    youtube_url = request.json.get('youtube_url')
    if not youtube_url:
        return jsonify({"error": "YouTube URL required"}), 400
    video_path = download_video(youtube_url)
    if not video_path:
        return jsonify({"error": "Failed to download video"}), 500
    processing = True
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    return jsonify({"message": "Processing started"})

@app.route('/stop', methods=['POST'])
def stop_processing():
    global processing
    processing = False
    return jsonify({"message": "Processing stopped"})

@app.route('/video_feed')
def video_feed():
    def generate():
        while processing:
            frame = process_stream()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)
