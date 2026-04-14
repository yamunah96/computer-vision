import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("Cars.mp4")
'''
`cv2.CAP_PROP_BUFFERSIZE` sets how many frames OpenCV keeps buffered internally from the video 
source. A smaller buffer means `VideoCapture.read()` returns newer frames faster and avoids lag 
from old buffered frames, but it does not make inference itself faster.
 It helps keep the displayed video closer to real-time.
'''
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

unique_id = set()
frame_skip = 5# skip every other frame
resize_scale = 0.5      # run inference on smaller frames

while True:
    for _ in range(frame_skip + 1):
        ret, frame = cap.read()
        if not ret:
            break
  

    h, w = frame.shape[:2]
    frame_small = cv2.resize(frame, (int(w * resize_scale), int(h * resize_scale)))

    results = model.track(frame_small, persist=True, classes=[2], conf=0.7)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.id is not None:
                unique_id.add(int(box.id[0].item()))

    annotations = results[0].plot()
    annotations = cv2.resize(annotations, (w, h))

    cv2.putText(annotations, f"Total Cars: {len(unique_id)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("YOLOv8 Detection", annotations)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(f"Total Cars: {len(unique_id)}")

cap.release()
cv2.destroyAllWindows()
print(len(unique_id))