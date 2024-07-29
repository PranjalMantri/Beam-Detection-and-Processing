import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

model = YOLO("src/models/beam_detector.pt")

image_path = "public/images/Class 1/PDF 3_1.png"
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

def calculate_intersection_over_union(rect1, rect2):
    x1, y1, width1, height1 = rect1
    x2, y2, width2, height2 = rect2

    x_intersection, y_intersection = max(x1, x2), max(y1, y2)
    x_intersection_end, y_intersection_end = min(x1 + width1, x2 + width2), min(y1 + height1, y2 + height2)

    intersection_width = x_intersection_end - x_intersection
    intersection_height = y_intersection_end - y_intersection

    if intersection_width <= 0 or intersection_height <= 0:
        return 0, 0

    intersection_area = intersection_width * intersection_height
    area_rect1 = width1 * height1
    area_rect2 = width2 * height2
    union_area = area_rect1 + area_rect2 - intersection_area

    iou = intersection_area / union_area
    return iou, intersection_width

def merge_rectangles(rect1, rect2):
    x1, y1, width1, height1 = rect1
    x2, y2, width2, height2 = rect2

    x_merged = min(x1, x2)
    y_merged = min(y1, y2)
    width_merged = max(x1 + width1, x2 + width2) - x_merged
    height_merged = max(y1 + height1, y2 + height2) - y_merged

    return (x_merged, y_merged, width_merged, height_merged)

beam_output_dir = 'public/Beams'
vertical_scale_output_dir = 'public/vertical_scales'

for output_dir in [beam_output_dir, vertical_scale_output_dir]:
    if os.path.isdir(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            removing = os.remove(file_path)
            if removing:
                print(f"Deleted file from {output_dir}")
            
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