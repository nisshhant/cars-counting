import cv2
import av
import threading
import time
import torch
from collections import deque
from ultralytics import YOLO

# RTSP Stream URL
rtsp_url = "rtsp://admin:Admin@123@192.168.1.202:554/profile1"
# Thread-safe deque to hold the latest frame
frame_deque = deque(maxlen=1)  # Only keep the most recent frame
frame_lock = threading.Lock()
line_y = 270  # Adjust based on your video
global entry_count, exit_count
entry_count = 0
exit_count = 0
vehicle_tracks = {}

# Define colors for visualization
BOX_COLOR = (0, 255, 0)  # Green for bounding boxes
TEXT_COLOR = (0, 255, 0)  # Green for text
LINE_COLOR = (0, 0, 255)  # Red for entry/exit line

# Class ID for 'car' in COCO dataset (YOLOv8 trained on COCO dataset)
CAR_CLASS_ID = 2

def capture_frames(rtsp_url):
    """Capture frames from RTSP stream with reconnect logic."""
    while True:
        try:
            container = av.open(rtsp_url, options={
                "rtsp_transport": "tcp",
                "fflags": "nobuffer",
                "max_delay": "500000"  # 0.5 seconds
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
    global exit_count, entry_count
    model = YOLO('yolov8n.pt')  # Load YOLOv8 model
    model.fuse()
    if torch.cuda.is_available():
        model.cuda()
    
    while True:
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
                    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
                    classes = result.boxes.cls.cpu().numpy()  # Get detected class IDs
                    
                    for box, track_id, cls_id in zip(boxes, ids, classes):
                        if int(cls_id) != CAR_CLASS_ID:
                            continue  # Only process cars
                        
                        x, y, w, h = box
                        center_y = y + h / 2  # Vehicle center position
                        
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
                        
                        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), BOX_COLOR, 2)
                        cv2.putText(img, f'ID {int(track_id)}', (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)
                
                cv2.line(img, (0, line_y), (img.shape[1], line_y), LINE_COLOR, 2)
                cv2.putText(img, f'Entry: {entry_count}  Exit: {exit_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Car Tracking", img)
            except Exception as e:
                print(f"Processing error: {str(e)}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


threading.Thread(target=capture_frames, args=(rtsp_url,), daemon=True).start()
process_stream()