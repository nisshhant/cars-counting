from ultralytics import YOLO
import cv2
import cvzone
from sort import *  # SORT tracker
import numpy as np
import pyodbc
import datetime

# Load YOLOv8 model
model = YOLO("YOLO-Weights/yolov8n.pt")

# Initialize video capture
cap = cv2.VideoCapture("E:\Internship\Pictures-Videos\object-detection.mp4")

# SORT tracker
tracker = Sort(max_age=10, min_hits=3)

# Counting line (Y-coordinate)
line_yl = 350
line_yr = 500
cars_count_up = 0  # Counter for cars going up
cars_count_down = 0  # Counter for cars going down
tracked_cars = {}  # Store each car's previous position
unique_cars = set()  # Store unique car IDs
stuck_cars = set()  # Cars that don't contribute to count change

# Database connection
try:
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-KHKO40I\\SQLEXPRESS;"
        "DATABASE=Car-count;"
        "Trusted_Connection=yes;"
    )
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


# Object class names
classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
              "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
              "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
              "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
              "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", 
              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    if not success:
        break

    # Run YOLO object detection
    results = model(img, stream=True)
    detections = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass == "car":  # Track only cars
                detections.append([x1, y1, x2, y2, conf])  # Format for SORT tracker
    
    # Update tracker with detections
    tracked_objects = tracker.update(np.array(detections))

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        w, h = x2 - x1, y2 - y1 
        cy = y1 + h // 2  # Midpoint of car

        # If a new car is detected, add it to the set
        if track_id not in unique_cars:
            unique_cars.add(track_id)
            tracked_cars[track_id] = cy
            stuck_cars.add(track_id)  # Initially mark car as stuck

        else:
            prev_cy = tracked_cars[track_id]  # Get previous Y position

            if prev_cy < line_yr and cy >= line_yr:  # Car moving downward
                cars_count_down += 1
                stuck_cars.discard(track_id)  # Remove from stuck cars
            elif prev_cy > line_yl and cy <= line_yl:  # Car moving upward
                cars_count_up += 1
                stuck_cars.discard(track_id)
            
            # Update car's last position
            tracked_cars[track_id] = cy

        # Draw bounding box & ID
        color = (0, 255, 0) if track_id not in stuck_cars else (0, 0, 255)  # Green if counted, red if stuck
        cvzone.putTextRect(img, f"Car {track_id}", (max(0, x1), max(30, y1)), scale=1, thickness=1, offset=3, colorR=color)
        cvzone.cornerRect(img, (x1, y1, w, h), 9, rt=5, colorC=color)

    # Draw counting line
    cv2.line(img, (0, line_yr), (img.shape[1]//2, line_yr), (0, 255, 0), 2)
    cv2.line(img, (img.shape[1]//2, line_yl), (img.shape[1], line_yl), (0, 255, 0), 2)
    cv2.rectangle(img, (45, 25), (45 + 250, 65), (0, 0, 0), -1)  # Background for Down counter
    cv2.rectangle(img, (45, 75), (45 + 220, 115), (0, 0, 0), -1)  # Background for Up counter
    cv2.putText(img, f"Cars Down: {cars_count_down}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f"Cars Up: {cars_count_up}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    update_database(cars_count_up, cars_count_down)
    print("Data updated successfully.")


    # Show results
    cv2.imshow("Car Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
