from flask import Flask, render_template, request, jsonify
import yt_dlp
import cv2
import av
import re
import threading
import time
import torch
from collections import deque
from ultralytics import YOLO
from Timer import Timer
import pyodbc
import datetime
# Global Variables
frame_deque = deque(maxlen=10)
frame_lock = threading.Lock()
line_y = 270  # Default line position
entry_count = 0
exit_count = 0
vehicle_tracks = {}
rtsp_url = ""  # User-defined camera path
capture_thread = None
streaming = False
cameraip=None
#def tick():
#    global cameraip,entry_count,exit_count
#    try:
#        timer.stop
#        data = { "CamIP": cameraip,"entry_count":entry_count, "exit_count": exit_count }
#        print(data)
#    except Exception as ex:
#        pass
#    finally:
#        timer.start()
#timer = Timer(3,tick)
#timer.start()
app = Flask(__name__)



def connectdb():

    try:
        global conn
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=DESKTOP-KHKO40I\\SQLEXPRESS;"
            "DATABASE=Car-count;"
            "Trusted_Connection=yes;"
        )
        global cursor
        cursor = conn.cursor()
        print("Connected to SQL Server successfully!")

    # Ensure the table exists (create it if not)
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Count_Data')
        BEGIN
        CREATE TABLE Count_Data (
            id INT IDENTITY(1,1) PRIMARY KEY,
            Entry_Count INT,
            Exit_Count INT,
            Timestamps TIME(7),
            Date DATE
        )
        END
        """)
        conn.commit()
        print("Table checked/created successfully.")

    except pyodbc.Error as e:
        print("Failed to connect to SQL Server.")
        print("Error:", e)
        exit()

def update_database(entry_count, exit_count):
    """Update the single record in Count_Data table with latest counts."""
    try:
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        # Check if there is an existing record
        cursor.execute("SELECT COUNT(*) FROM Count_Data")
       
        record_count = cursor.fetchone()[0]
        if record_count == 0:
            # If no record exists, insert a new one
            cursor.execute("""
                INSERT INTO Count_Data (Entry_Count, Exit_Count, Timestamps, Date)
                VALUES (?, ?, ?, ?)
            """, (entry_count, exit_count, current_time, current_date))
        else:
            # If record exists, update it
            cursor.execute("""
                UPDATE Count_Data
                SET Entry_Count = ?, Exit_Count = ?, Timestamps = ?, Date = ?
            """, (entry_count, exit_count, current_time, current_date))

        conn.commit()
        print("Database record updated successfully.")

    except pyodbc.Error as e:
        print("Database update failed:", e)

# Load YOLO model
model = YOLO('yolov8n.pt')
model.fuse()
if torch.cuda.is_available():
    model.cuda()

def capture_frames():
    global streaming
    while streaming:
        try:
            container = av.open(rtsp_url, options={
                "rtsp_transport": "tcp",
                "fflags": "nobuffer",
                "max_delay": "500000"
            })
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.resize(img, (640, 480))
                with frame_lock:
                    frame_deque.append(img)
        except Exception as e:
            print(f"Capture error: {str(e)} - Reconnecting...")
            time.sleep(5)
def process_stream():
    global exit_count, entry_count, streaming
    while streaming:
        img = None
        with frame_lock:
            if frame_deque:
                img = frame_deque.pop()
        if img is not None:
            try:
                results = model.track(img, persist=True, verbose=False, imgsz=640)
                for result in results:
                    if result.boxes is None:
                        continue
                    boxes = result.boxes.xywh.cpu().numpy()
                    print(boxes)
                    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
                    classes = result.boxes.cls.cpu().numpy()
                    for box, track_id, cls_id in zip(boxes, ids, classes):
                        if int(cls_id) != 2:
                            continue
                        x, y, w, h = box
                        center_x, center_y = x, y  # Center point (x, y)
                        
                        if track_id not in vehicle_tracks:
                            vehicle_tracks[track_id] = []
                        vehicle_tracks[track_id].append(center_y)

                        if len(vehicle_tracks[track_id]) > 2:
                            prev_y = vehicle_tracks[track_id][-2]
                            curr_y = vehicle_tracks[track_id][-1]
                            if prev_y < line_y and curr_y >= line_y:
                                entry_count += 1
                            elif prev_y > line_y and curr_y <= line_y:
                                exit_count += 1
                        
                        connectdb()
                        update_database(entry_count, exit_count)

                        # Draw rectangle around the detected object
                        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), 
                                      (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                        
                        # Display ID and coordinates
                        cv2.putText(img, f'ID {int(track_id)}', 
                                    (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (0, 255, 0), 2)
                        cv2.putText(img, f'Center: ({int(center_x)}, {int(center_y)})', 
                                    (int(x), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (255, 255, 0), 2)
                        cv2.putText(img, f'W: {int(w)} H: {int(h)}', 
                                    (int(x), int(y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (0, 255, 255), 2)

                # Draw the counting line
                cv2.line(img, (0, line_y), (img.shape[1], line_y), (0, 0, 255), 2)

                # Display entry and exit counts
                cv2.putText(img, f'Entry: {entry_count}  Exit: {exit_count}', 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                _, buffer = cv2.imencode('.jpg', img)
                return buffer.tobytes()

            except Exception as e:
                print(f"Processing error: {str(e)}")
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_stream():
    global rtsp_url, capture_thread, streaming,cameraip
    rtsp_url = request.json.get('rtsp_url')
    match = re.search(r'(\d+\.\d+\.\d+\.\d+)', rtsp_url)
    if match:
        ip_address = match.group(1)
        cameraip=ip_address
        print("Extracted IP Address:", ip_address)
    else:
        print("No IP address found")
    if not rtsp_url:
        return jsonify({"error": "RTSP URL required"}), 400
    streaming = True
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    return jsonify({"message": "Stream started"})

@app.route('/stop', methods=['POST'])
def stop_stream():
    global streaming
    streaming = False
    return jsonify({"message": "Stream stopped"})

@app.route('/ResetCount', methods=['POST'])
def ResetCount():
    global entry_count,exit_count
    entry_count = 0
    exit_count = 0
    return jsonify({"message": "Reset"})


@app.route('/update_line', methods=['POST'])
def update_line():
    global line_y
    line_y = int(request.json.get('line_y', 270))
    return jsonify({"message": f"Line set to {line_y}"})

@app.route('/video_feed')
def video_feed():
    global cameraip,entry_count,exit_count
    def generate():
        while streaming:
            frame = process_stream()
            data = { "CamIP": cameraip,"entry_count":entry_count, "exit_count": exit_count }
            print(data)
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return app.response_class(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

   
if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=5004, debug=True)