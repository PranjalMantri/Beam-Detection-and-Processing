import cv2
from ultralytics import YOLO
import os
import warnings
import re
from src.detection.helper.plot_detections import plot_one_box

warnings.filterwarnings("ignore", category=FutureWarning)

# To get file numbers of existing files to maintain a consister naming convention
def get_next_file_number(directory):
    existing_files = os.listdir(directory)
    numbers = [int(re.search(r'vertical_scale_(\d+)', f).group(1)) for f in existing_files if re.search(r'vertical_scale_(\d+)', f)]
    return max(numbers) + 1 if numbers else 0

def detect_and_save_vertical(image_path, model_path="src/models/scale_detector.pt", output_dir="public/vertical_scales", add_padding=False):

    model = YOLO(model_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)
    detections = results[0].boxes.data.cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)

    vertical_scales = []

    for idx, (*xyxy, conf, cls) in enumerate(detections):
        xmin, ymin, xmax, ymax = map(int, xyxy)
        width, height = xmax - xmin, ymax - ymin
        
        # Assume a scale is vertical if its height is greater than its width
        if height > width:
            vertical_scales.append((xmin, ymin, xmax, ymax))

    next_file_number = get_next_file_number(output_dir)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(vertical_scales, start=next_file_number):
        # Don't need to add padding in vertical scale generally
        if add_padding:
            padding = 30
            xmin = max(0, xmin - padding)
            xmax = min(image_rgb.shape[1], xmax + padding)
            ymin = max(0, ymin - padding)
            ymax = min(image_rgb.shape[0], ymax + padding)
  
        rect_image = image_rgb[ymin:ymax, xmin:xmax]

        rect_image_path = os.path.join(output_dir, f'vertical_scale_{idx}.png')
        cv2.imwrite(rect_image_path, cv2.cvtColor(rect_image, cv2.COLOR_RGB2BGR))

        plot_one_box([xmin, ymin, xmax, ymax], image_rgb, color=(0, 255, 0), line_thickness=2)

    output_path = os.path.join(output_dir, f'vertical_scale_annotated_{next_file_number}.png')
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    return True

if __name__ == "__main__":
    image_path = "public/images/Class 1/PDF 3_1.png"
    model_path = "src/models/scale_detector.pt"
    output_dir = "public/vertical_scales"
    detect_and_save_vertical(image_path, model_path, output_dir, add_padding=True)
