import cv2
import time
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# Initialize YOLO model
model = YOLO("fish2.pt")

# Video input
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Screen dimensions
screen_width = 1280
screen_height = 720

# Center line for counting
center_line = screen_width // 2

# Initialize tracking data
fish_tracks = {}
fish_counts = {"left": {}, "right": {}}

# Function to initialize a Kalman filter for tracking
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],  
                      [0, 1, 0, 1],  
                      [0, 0, 1, 0],  
                      [0, 0, 0, 1]])  # State transition
    kf.H = np.array([[1, 0, 0, 0],  
                      [0, 1, 0, 0]])  # Measurement function
    kf.P *= 1000  # Initial uncertainty
    kf.R = np.array([[10, 0], [0, 10]])  # Measurement noise
    kf.Q = np.eye(4) * 0.1  # Process noise
    return kf

# Function to predict and detect fish
def predict_and_detect(chosen_model, img, conf=0.5):
    results = chosen_model.predict(img, conf=conf)
    detected_fish = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2  # Fish center x
            cy = (y1 + y2) // 2  # Fish center y
            detected_fish.append((cx, cy, class_id))
    return img, detected_fish

prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.resize(img, (screen_width, screen_height))
    result_img, detected_fish = predict_and_detect(model, img, conf=0.5)
    
    # Update Kalman Filters
    updated_tracks = {}
    for cx, cy, class_id in detected_fish:
        found_match = False
        for track_id, (kf, last_class, last_pos) in fish_tracks.items():
            kf.predict()
            px, py = kf.x[:2]  # Predicted position
            if abs(px - cx) < 50 and abs(py - cy) < 50:  # Match detected
                kf.update([cx, cy])
                updated_tracks[track_id] = (kf, class_id, (cx, cy))
                found_match = True
                break
        
        if not found_match:  # Create new track
            new_kf = create_kalman_filter()
            new_kf.x[:2] = np.array([cx, cy]).reshape((2, 1))
            updated_tracks[len(updated_tracks)] = (new_kf, class_id, (cx, cy))
    
    # Count fish crossing the center line
    for track_id, (kf, class_id, (cx, cy)) in updated_tracks.items():
        prev_x, prev_y = fish_tracks.get(track_id, (kf, class_id, (cx, cy)))[2]
        if prev_x < center_line and cx >= center_line:
            fish_counts["right"].setdefault(class_id, 0)
            fish_counts["right"][class_id] += 1
            print(f"Fish {model.names[class_id]} crossed to the right. Total: {fish_counts['right'][class_id]}")
        elif prev_x > center_line and cx <= center_line:
            fish_counts["left"].setdefault(class_id, 0)
            fish_counts["left"][class_id] += 1
            print(f"Fish {model.names[class_id]} crossed to the left. Total: {fish_counts['left'][class_id]}")
    
    fish_tracks = updated_tracks
    
    # Draw fish tracking and counting
    for track_id, (kf, class_id, (cx, cy)) in fish_tracks.items():
        color = (0, 255, 0)
        cv2.circle(result_img, (cx, cy), 5, color, -1)
        cv2.putText(result_img, f"ID {track_id} {model.names[class_id]}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw center line
    cv2.line(result_img, (center_line, 0), (center_line, screen_height), (255, 0, 0), 2)
    
    # Display count
    y_offset = 50
    for direction, counts in fish_counts.items():
        for class_id, count in counts.items():
            text = f"{model.names[class_id]} {direction}: {count}"
            cv2.putText(result_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(result_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Image", result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print final counts
print("Fish counts:")
for direction, counts in fish_counts.items():
    for class_id, count in counts.items():
        print(f"{model.names[class_id]} {direction}: {count}")