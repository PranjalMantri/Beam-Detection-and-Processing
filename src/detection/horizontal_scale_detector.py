import cv2
from ultralytics import YOLO
import os
import warnings
from src.detection.helper.plot_detections import plot_one_box

warnings.filterwarnings("ignore", category=FutureWarning)

def detect_and_save_horizontal(image_path, model_path="src/models/scale_detector.pt", output_dir="public/horizontal_scales", add_padding=False):

    model = YOLO(model_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect objects in the image using the YOLO model
    results = model(image_rgb)
    detections = results[0].boxes.data.cpu().numpy()

    # Clear the output directory before saving new images
    if os.path.isdir(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            os.remove(file_path)

    os.makedirs(output_dir, exist_ok=True)

    horizontal_scales = []

    for idx, (*xyxy, conf, cls) in enumerate(detections):
        xmin, ymin, xmax, ymax = map(int, xyxy)
        width, height = xmax - xmin, ymax - ymin
        
        # Identify horizontal scales based on width being greater than height
        if width > height:
            horizontal_scales.append((xmin, ymin, xmax, ymax))

    for idx, (xmin, ymin, xmax, ymax) in enumerate(horizontal_scales):
        if add_padding:
            padding = 30
            xmin = max(0, xmin - padding)
            xmax = min(image_rgb.shape[1], xmax + padding)
        
        rect_image = image_rgb[ymin:ymax, xmin:xmax]

        rect_image_path = os.path.join(output_dir, f'horizontal_scale_{idx}.png')
        cv2.imwrite(rect_image_path, cv2.cvtColor(rect_image, cv2.COLOR_RGB2BGR))

        # Annotate the original image with bounding boxes
        plot_one_box([xmin, ymin, xmax, ymax], image_rgb, color=(255, 0, 0), line_thickness=2)

    # Save the annotated image showing all detected horizontal scales
    output_path = os.path.join(output_dir, 'horizontal_scale_annotated.png')
    cv2.imwrite(output_path, cv2.COLOR_RGB2BGR)

    return True

if __name__ == "__main__":
    image_path = "public/images/Class 1/PDF 3_1.png"
    model_path = "src/models/scale_detector.pt"
    output_dir = "public/horizontal_scales"
    detect_and_save_horizontal(image_path, model_path, output_dir, add_padding=True)
