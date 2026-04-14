import cv2
from ultralytics import YOLO


# Load a pretrained YOLO8n model
model = YOLO("yolov8n.pt")
cap= cv2.VideoCapture("v1.mp4")  

unique_id=set()
while True:
    ret,frame=cap.read()
    print("orginal frame size:",frame.shape)

    #  fix the video frames windows size by resizing the frame
    frame=cv2.resize(frame,(640,640))
    print("resized frame size:",frame.shape)

    if not ret:
        break

    results = model(frame)
    # print(results)

    for result in results:
        boxes=result.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            conf=box.conf[0]
            cls=box.cls[0]
            unique_id.add(cls.item())
    annotations = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotations) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

