import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

model = YOLO("src/models/scale_detector.pt")

# Single image path to detect horizontal scales
image_path = "public/images/Class 1/PDF 3_1.png"
output_dir = "public/horizontal_scales"

def detect_and_save_horizontal(image_path, model, output_dir, add_padding=False):
    print(f"Processing image: {image_path}")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)
    detections = results[0].boxes.data.cpu().numpy()

    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
        color = color or [255, 0, 0]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    if os.path.isdir(output_dir):
        print(f"Clearing output directory: {output_dir}")
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            os.remove(file_path)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory prepared: {output_dir}")

    horizontal_scales = []

    for idx, (*xyxy, conf, cls) in enumerate(detections):
        xmin, ymin, xmax, ymax = map(int, xyxy)
        width, height = xmax - xmin, ymax - ymin
        
        # Assume a scale is horizontal if its width is greater than its height
        if width > height:
            horizontal_scales.append((xmin, ymin, xmax, ymax))

    for idx, (xmin, ymin, xmax, ymax) in enumerate(horizontal_scales):
        if add_padding:
            padding = 30
            xmin = max(0, xmin - padding)
            xmax = min(image_rgb.shape[1], xmax + padding)
            # ymin and ymax remain unchanged to add padding only on left and right sides
        
        rect_image = image_rgb[ymin:ymax, xmin:xmax]

        rect_image_path = os.path.join(output_dir, f'horizontal_scale_{idx}.png')
        cv2.imwrite(rect_image_path, cv2.cvtColor(rect_image, cv2.COLOR_RGB2BGR))
        print(f"Saved horizontal scale image: {rect_image_path}")

        plot_one_box([xmin, ymin, xmax, ymax], image_rgb, color=(255, 0, 0), line_thickness=2)

    output_path = os.path.join(output_dir, 'horizontal_scale_annotated.png')
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image with horizontal scales: {output_path}")

# Detect and save horizontal scales from the single image
detect_and_save_horizontal(image_path, model, output_dir, add_padding=True)
