import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import warnings
import re

warnings.filterwarnings("ignore", category=FutureWarning)

model = YOLO("src/models/scale_detector.pt")

# Single image path to detect vertical scales
image_path = "public/images/Class 1/PDF 3_1.png"
output_dir = "public/vertical_scales"

def get_next_file_number(directory):
    existing_files = os.listdir(directory)
    numbers = [int(re.search(r'vertical_scale_(\d+)', f).group(1)) for f in existing_files if re.search(r'vertical_scale_(\d+)', f)]
    return max(numbers) + 1 if numbers else 0

def detect_and_save_vertical(image_path, model, output_dir, add_padding=False):
    print(f"Processing image: {image_path}")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)
    detections = results[0].boxes.data.cpu().numpy()

    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
        color = color or [0, 255, 0]  # Changed to green for vertical scales
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory prepared: {output_dir}")

    vertical_scales = []

    for idx, (*xyxy, conf, cls) in enumerate(detections):
        xmin, ymin, xmax, ymax = map(int, xyxy)
        width, height = xmax - xmin, ymax - ymin
        
        # Assume a scale is vertical if its height is greater than its width
        if height > width:
            vertical_scales.append((xmin, ymin, xmax, ymax))

    next_file_number = get_next_file_number(output_dir)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(vertical_scales, start=next_file_number):
        if add_padding:
            padding = 30
            ymin = max(0, ymin - padding)
            ymax = min(image_rgb.shape[0], ymax + padding)
            # xmin and xmax remain unchanged to add padding only on top and bottom
        
        rect_image = image_rgb[ymin:ymax, xmin:xmax]

        rect_image_path = os.path.join(output_dir, f'vertical_scale_{idx}.png')
        cv2.imwrite(rect_image_path, cv2.cvtColor(rect_image, cv2.COLOR_RGB2BGR))
        print(f"Saved vertical scale image: {rect_image_path}")

        plot_one_box([xmin, ymin, xmax, ymax], image_rgb, color=(0, 255, 0), line_thickness=2)

    output_path = os.path.join(output_dir, f'vertical_scale_annotated_{next_file_number}.png')
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image with vertical scales: {output_path}")

# Detect and save vertical scales from the single image
detect_and_save_vertical(image_path, model, output_dir, add_padding=True)