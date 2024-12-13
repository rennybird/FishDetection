import cv2
import time
from ultralytics import YOLO

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def get_color_for_class(class_id):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    return colors[class_id % len(colors)]

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            color = get_color_for_class(class_id)
            confidence = box.conf[0]
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), color, rectangle_thickness)
            cv2.putText(img, f"{result.names[class_id]} {confidence:.2f}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, text_thickness)
    return img, results

model = YOLO("fish1.pt")

video_path = r"test.mp4"
cap = cv2.VideoCapture(video_path)

screen_width = 1280
screen_height = 720

prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Resize the frame to fit the screen
    img = cv2.resize(img, (screen_width, screen_height))
    
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Display FPS on the frame
    cv2.putText(result_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Image", result_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
