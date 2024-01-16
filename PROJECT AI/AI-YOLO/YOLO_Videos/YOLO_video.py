from ultralytics import YOLO
import cv2
import math
from datetime import datetime

cap = cv2.VideoCapture('../Videos/busfinal.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the output video parameters
output_fps = 15  # Adjust the output frames per second as needed
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), output_fps, (frame_width, frame_height))

model = YOLO('yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "safety mask", "face shield", "hand sanitizer", "glasses",
              ]

# Initialize total passenger count and set of persons who crossed the line
total_passenger_count = 0
persons_crossed_line = set()

# Define the line coordinates (center of the frame) and line thickness
line_start = (frame_width // 4, frame_height // 2)
line_end = (frame_width * 3 // 4, frame_height // 2)
line_thickness = 5

while True:
    success, img = cap.read()

    if not success:
        break

    # Draw the line on the frame with increased thickness
    cv2.line(img, line_start, line_end, (0, 0, 255), line_thickness)

    # Display date and time in front of the total passenger count
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, current_time, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Initialize passenger count for the current frame
    passenger_count = 0

    # Doing detections using YOLOv8 frame by frame
    results = model(img, stream=True)

    # Once we have the results, loop through them and count the number of specific class detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Draw rectangle around detected object with a customized color
            color = (0, 255, 0) if class_name == "person" else (255, 0, 255)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # Draw class name on the frame with customized text properties
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Update passenger count based on the detected class
            if class_name == "person":
                passenger_count += 1

                # Check if the person crosses the line and not already counted
                if y2 >= line_start[1] and y1 <= line_start[1] and (x1, y1, x2, y2) not in persons_crossed_line:
                    persons_crossed_line.add((x1, y1, x2, y2))
                    total_passenger_count += 1

                    # Mark the person who crossed the line on the frame
                    cv2.putText(img, 'Person Crossed the Line', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw passenger count on the frame with a customized color and text properties
    cv2.putText(img, f'Passenger Count: {passenger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw total passenger count on the frame with a customized color and text properties
    cv2.putText(img, f'Total Passenger Count: {total_passenger_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
