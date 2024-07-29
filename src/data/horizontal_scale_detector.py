import cv2
from ultralytics import YOLO
import os
import warnings
from helper.plot_detections import plot_one_box

warnings.filterwarnings("ignore", category=FutureWarning)

def detect_and_save_horizontal(image_path, model_path, output_dir, add_padding=False):
    print(f"Processing image: {image_path}")

    model = YOLO(model_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)
    detections = results[0].boxes.data.cpu().numpy()

    # Delete all the existing images and store newer detected ones
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
        # Adding padding horizontally for better processing later
        if add_padding:
            padding = 30
            xmin = max(0, xmin - padding)
            xmax = min(image_rgb.shape[1], xmax + padding)
        
        rect_image = image_rgb[ymin:ymax, xmin:xmax]

        rect_image_path = os.path.join(output_dir, f'horizontal_scale_{idx}.png')
        cv2.imwrite(rect_image_path, cv2.cvtColor(rect_image, cv2.COLOR_RGB2BGR))
        print(f"Saved horizontal scale image: {rect_image_path}")

        plot_one_box([xmin, ymin, xmax, ymax], image_rgb, color=(255, 0, 0), line_thickness=2)

    output_path = os.path.join(output_dir, 'horizontal_scale_annotated.png')
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image with horizontal scales: {output_path}")

if __name__ == "__main__":
    image_path = "public/images/Class 1/PDF 3_1.png"
    model_path = "src/models/scale_detector.pt"
    output_dir = "public/horizontal_scales"
    detect_and_save_horizontal(image_path, model_path, output_dir, add_padding=True)
