import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time
from math import dist

# Load YOLO model and initialize the tracker
model = YOLO('yolov8s.pt')
tracker = Tracker()

# Set up video capture and create a window for displaying the results
cv2.namedWindow('Vehicle Detection And Tracking')
cap = cv2.VideoCapture('veh2.mp4')

# Read the class labels from a file
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

# Define the crossing lines and offset for vehicle counting
crossing_line_y1 = 322
crossing_line_y2 = 368
offset = 6

# Initialize dictionaries for storing vehicle timestamps and counters
vehicles_down = {}
counter_down = []
vehicles_up = {}
counter_up = []
speeds = {}

# Main loop for processing each frame of the video
while True:
    # Read the frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection using YOLO model
    results = model.predict(frame)
    boxes = results[0].boxes.data

    vehicle_list = []
    # Extract bounding box coordinates and class labels for vehicles
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        class_id = int(box[5])
        class_name = class_list[class_id]
        if 'car' in class_name:
            vehicle_list.append([x1, y1, x2, y2])

    # Update the tracker with the current vehicle list
    bbox_ids = tracker.update(vehicle_list)

    # Process each tracked vehicle
    for bbox_id in bbox_ids:
        x1, y1, x2, y2, vehicle_id = bbox_id
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Draw a bounding box around the vehicle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Check if vehicle crosses the lower line
        if crossing_line_y1 - offset < center_y < crossing_line_y1 + offset:
            vehicles_down[vehicle_id] = time.time()

        # Check if vehicle crosses the upper line after crossing the lower line
        if vehicle_id in vehicles_down:
            if crossing_line_y2 - offset < center_y < crossing_line_y2 + offset:
                elapsed_time = time.time() - vehicles_down[vehicle_id]
                if vehicle_id not in counter_up:
                    counter_up.append(vehicle_id)
                    distance = 10  # meters
                    speed_ms = distance / elapsed_time
                    speed_kph = speed_ms * 3.6
                    speeds[vehicle_id] = speed_kph

        # Check if vehicle crosses the upper line
        if crossing_line_y2 - offset < center_y < crossing_line_y2 + offset:
            vehicles_up[vehicle_id] = time.time()

        # Check if vehicle crosses the lower line after crossing the upper line
        if vehicle_id in vehicles_up:
            if crossing_line_y1 - offset < center_y < crossing_line_y1 + offset:
                elapsed_time = time.time() - vehicles_up[vehicle_id]
                if vehicle_id not in counter_down:
                    counter_down.append(vehicle_id)
                    distance = 10  # meters
                    speed_ms = distance / elapsed_time
                    speed_kph = speed_ms * 3.6
                    speeds[vehicle_id] = speed_kph

        # Display the speed for each vehicle
        if vehicle_id in speeds:
            speed_text = str(int(speeds[vehicle_id])) + ' Km/h'
            cv2.putText(frame, speed_text, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Draw crossing lines and display vehicle counts
    cv2.line(frame, (274, crossing_line_y1), (814, crossing_line_y1), (255, 255, 255), 1)
    cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (177, crossing_line_y2), (927, crossing_line_y2), (255, 255, 255), 1)
    cv2.putText(frame, 'L2', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    num_vehicles_down = len(counter_up)
    num_vehicles_up = len(counter_down)
    cv2.putText(frame, 'goingdown: ' + str(num_vehicles_down), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, 'goingup: ' + str(num_vehicles_up), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Display the frame with vehicle detection and tracking results
    cv2.imshow("Vehicle Detection And Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
