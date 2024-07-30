import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import warnings
from src.detection.helper.nms import calculate_intersection_over_union, merge_rectangles
from src.detection.helper.plot_detections import plot_one_box

warnings.filterwarnings("ignore", category=FutureWarning)

def detect_beams(image_path, model_path="src/models/beam_detector.pt"):
    model = YOLO(model_path)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)
    detections = results[0].boxes.data.cpu().numpy()

    beam_output_dir = 'public/Beams'
    vertical_scale_output_dir = 'public/vertical_scales'

    # If directory already exists, delete it to remove previous images
    for output_dir in [beam_output_dir, vertical_scale_output_dir]:
        if os.path.isdir(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                os.remove(file_path)
        os.makedirs(output_dir, exist_ok=True)

    beams = []
    vertical_scales = []

    for idx, (*xyxy, conf, cls) in enumerate(detections):
        xmin, ymin, xmax, ymax = map(int, xyxy)
        width, height = xmax - xmin, ymax - ymin
        rect = (xmin, ymin, width, height)
        
        # Classify as beam or vertical scale based on aspect ratio
        if width > height:
            target_list = beams
        else:
            target_list = vertical_scales
        
        merged = False
        for i, existing_rect in enumerate(target_list):
            iou, _ = calculate_intersection_over_union(rect, existing_rect)
            if iou > 0.5:
                target_list[i] = merge_rectangles(rect, existing_rect)
                merged = True
                break
        
        if not merged:
            target_list.append(rect)

    # Save beams
    for idx, (xmin, ymin, width, height) in enumerate(beams):
        xmax, ymax = xmin + width, ymin + height
        rect_image = image_rgb[ymin:ymax, xmin:xmax]
        
        rect_image_path = os.path.join(beam_output_dir, f'beam_{idx}.png')
        cv2.imwrite(rect_image_path, cv2.cvtColor(rect_image, cv2.COLOR_RGB2BGR))
        
        plot_one_box([xmin, ymin, xmax, ymax], image_rgb, color=(255, 0, 0), label="Beam", line_thickness=2)

    # Save vertical scales
    for idx, (xmin, ymin, width, height) in enumerate(vertical_scales):
        xmax, ymax = xmin + width, ymin + height
        rect_image = image_rgb[ymin:ymax, xmin:xmax]
        
        rect_image_path = os.path.join(vertical_scale_output_dir, f'vertical_scale_{idx}.png')
        cv2.imwrite(rect_image_path, cv2.cvtColor(rect_image, cv2.COLOR_RGB2BGR))
        
        plot_one_box([xmin, ymin, xmax, ymax], image_rgb, color=(0, 255, 0), label="Vertical Scale", line_thickness=2)

    output_path = 'public/beams_and_scales.png'
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    print(f"Detected and saved {len(beams)} beams and {len(vertical_scales)} vertical scales.")
    print(f"Annotated image saved as {output_path}")

    return True

if __name__ == "__main__":
    image_path = "public/images/Class 1/PDF 3_1.png"
    # model_path = "src/models/beam_detector.pt"
    detect_beams(image_path)
