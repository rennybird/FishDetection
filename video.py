import cv2
import time
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linear_sum_assignment

# Initialize YOLO model
model = YOLO("fish3.pt")

# Video input
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Screen dimensions
screen_width = 1080
screen_height = 720

# Center line for counting
center_line = screen_width // 2

# Initialize tracking data
fish_tracks = {}
fish_counts = {"left": {}, "right": {}}
fish_history = {}  # To store history of detected fish types
fish_directions = {}  # To store history of fish directions
lost_threshold = 50  # Number of frames to keep a track before considering it lost

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

def predict_and_detect(chosen_model, img, conf=0.3, iou=0.3):
    results = chosen_model.predict(img, conf=conf, iou=iou)
    detected_fish = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2  # Fish center x
            cy = (y1 + y2) // 2  # Fish center y
            detected_fish.append((cx, cy, class_id))
    return img, detected_fish

def update_tracks(detected_fish):
    updated_tracks = {}
    if detected_fish:
        # Create cost matrix
        num_tracks = len(fish_tracks)
        num_detections = len(detected_fish)
        cost_matrix = np.full((num_tracks, num_detections), np.inf)  # Initialize with high values
        
        track_ids = list(fish_tracks.keys())
        for i, track_id in enumerate(track_ids):
            kf, last_class, last_pos, lost_frames = fish_tracks[track_id]
            kf.predict()
            px, py = kf.x[:2]  # Predicted position
            
            for j, (cx, cy, class_id) in enumerate(detected_fish):
                # Calculate distance cost
                distance_cost = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                
                # Add penalty for class mismatch
                class_penalty = 500 if class_id != last_class else 0  # Increase penalty to prioritize correct class
                
                cost_matrix[i, j] = distance_cost + class_penalty

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_detections = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 100:  # Threshold for valid assignments
                track_id = track_ids[i]
                cx, cy, class_id = detected_fish[j]
                kf, _, _, _ = fish_tracks[track_id]
                kf.update([cx, cy])
                updated_tracks[track_id] = (kf, class_id, (cx, cy), 0)  # Reset lost_frames
                assigned_detections.add(j)

        # Create new tracks for unassigned detections
        for j, (cx, cy, class_id) in enumerate(detected_fish):
            if j not in assigned_detections:
                new_kf = create_kalman_filter()
                new_kf.x[:2] = np.array([cx, cy]).reshape((2, 1))
                new_track_id = f"{class_id}_{len(fish_tracks)}"
                updated_tracks[new_track_id] = (new_kf, class_id, (cx, cy), 0)  # Initialize lost_frames
                fish_history[new_track_id] = deque(maxlen=2)  
                fish_directions[new_track_id] = deque(maxlen=2)  

    # Handle lost tracks
    for track_id, (kf, class_id, last_pos, lost_frames) in fish_tracks.items():
        if track_id not in updated_tracks:
            if lost_frames < lost_threshold:
                updated_tracks[track_id] = (kf, class_id, last_pos, lost_frames + 1)
            else:
                # Track is considered lost after threshold
                pass

    return updated_tracks


def update_fish_history(updated_tracks):
    for track_id, (kf, class_id, (cx, cy), lost_frames) in updated_tracks.items():
        fish_history[track_id].append(class_id)
        most_common_class_id, count = Counter(fish_history[track_id]).most_common(1)[0]
        if count > len(fish_history[track_id]) // 2:
            updated_tracks[track_id] = (kf, most_common_class_id, (cx, cy), lost_frames)
        prev_x, prev_y = fish_tracks.get(track_id, (kf, class_id, (cx, cy), lost_frames))[2]
        direction = "right" if cx > prev_x else "left"
        fish_directions[track_id].append(direction)

def count_fish_crossing(updated_tracks):
    for track_id, (kf, class_id, (cx, cy), lost_frames) in updated_tracks.items():
        prev_x, prev_y = fish_tracks.get(track_id, (kf, class_id, (cx, cy), lost_frames))[2]
        if len(fish_directions[track_id]) == fish_directions[track_id].maxlen:
            consistent_direction = Counter(fish_directions[track_id]).most_common(1)[0][0]
            if consistent_direction == "right" and prev_x < center_line and cx >= center_line:
                fish_counts["right"].setdefault(class_id, 0)
                fish_counts["right"][class_id] += 1
                print(f"Fish {model.names[class_id]} crossed to the right. Total: {fish_counts['right'][class_id]}")
            elif consistent_direction == "left" and prev_x > center_line and cx <= center_line:
                fish_counts["left"].setdefault(class_id, 0)
                fish_counts["left"][class_id] += 1
                print(f"Fish {model.names[class_id]} crossed to the left. Total: {fish_counts['left'][class_id]}")

def draw_tracking_info(result_img, updated_tracks):
    for track_id, (kf, class_id, (cx, cy), lost_frames) in updated_tracks.items():
        color = (0, 255, 0)
        cv2.circle(result_img, (cx, cy), 5, color, -1)
        cv2.putText(result_img, f"ID {track_id} {model.names[class_id]}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        px, py = kf.x[:2]
        cv2.circle(result_img, (int(px), int(py)), 5, (0, 0, 255), -1)  # Red circle for prediction

def draw_center_line(result_img):
    cv2.line(result_img, (center_line, 0), (center_line, screen_height), (255, 0, 0), 2)

def display_counts(result_img):
    y_offset = 50
    for direction, counts in fish_counts.items():
        for class_id, count in counts.items():
            text = f"{model.names[class_id]} {direction}: {count}"
            cv2.putText(result_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return fps, current_time

prev_time = 0

def process_frame(img):
    img = cv2.resize(img, (screen_width, screen_height))
    result_img, detected_fish = predict_and_detect(model, img, conf=0.6, iou=0.6)  # Increased thresholds
    
    updated_tracks = update_tracks(detected_fish)
    update_fish_history(updated_tracks)
    count_fish_crossing(updated_tracks)
    
    global fish_tracks
    fish_tracks = updated_tracks  # Update the global fish_tracks variable
    
    draw_tracking_info(result_img, updated_tracks)
    draw_center_line(result_img)
    display_counts(result_img)
    
    return result_img

with ThreadPoolExecutor(max_workers=2) as executor:
    while True:
        success, img = cap.read()
        if not success:
            break
        
        future = executor.submit(process_frame, img)
        result_img = future.result()
        
        fps, prev_time = calculate_fps(prev_time)
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