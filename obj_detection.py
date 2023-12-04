import cv2
from ultralytics import YOLO
import os

def center_point(x_l, x_r, y_t, y_b):
  x = (x_r - x_l) * 0.5 + x_l
  y = (y_t - y_b) * 0.5 + y_b
  return (x, y)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Directory containing the PNG images
image_dir = "D:/KTH/IL2232&II2211_P5/kitti_dataset/testing/image_02/0002/"

# List all PNG files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

i = 0
for image_file in image_files:
    result_file = f'0002_{i}.txt'
    result_pth = os.path.join(output_dir, result_file)
    # Load the PNG image
    image_path = os.path.join(image_dir, image_file)
    frame = cv2.imread(image_path)

    # Run YOLOv8 tracking on the image, persisting tracks between frames
    results = model.track(frame,persist=True)

    # Visualize the results on the image
    annotated_frame = results[0].plot()
    boxes_info = results[0].boxes
    boxes_info.id
    print(i)
    with open(result_pth, 'w') as file:
        for obj in range(len(boxes_info.cls)):
            cls_label = float(boxes_info.cls[obj])
            if cls_label==0.0 or cls_label==1.0 or cls_label==2.0 or cls_label==3.0 or cls_label==5.0:
                x_l, y_t, x_r, y_b = boxes_info.xyxy[obj].tolist()
                x_center, y_center = center_point(x_l, x_r, y_t, y_b)
                try:
                    id = boxes_info.id[obj]
                except TypeError:
                    print("No object is tracked!!!")
                else:
                    file.write(f'{cls_label},{x_l},{y_t},{x_r},{y_b},{x_center},{y_center},{id}\n')

    
    # Display the annotated image
    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    i+=1
    # Wait for a key press (you can customize this behavior)
    key = cv2.waitKey(0)
    if key == 27:  # 'Esc' key
        break

# Close the display window
cv2.destroyAllWindows()
