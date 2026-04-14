
# python -m venv myvenv
# myevnv/Scripts/activate
#  python -m pip3 install --upgrade pip
#  pip install ultralytics,opencv-python
import cv2
from ultralytics import YOLO


# Load a pretrained YOLO8n model
model = YOLO("yolo8n.pt")

image= cv2.imread("cat.png")

results = model(image)
print(results)

annotations = results[0].plot()
cv2.imshow("YOLOv8 Detection", annotations) 
cv2.waitKey(0)
cv2.destroyAllWindows()
