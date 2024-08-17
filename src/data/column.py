import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.data.helper.line_processing import *


def merge_lines(lines, vertical_threshold=5, horizontal_threshold=20):
    if not lines:
        return []

    # Sort lines by y-coordinate (lines at top are first)
    lines = sorted(lines, key=lambda line: (line[1] + line[3]) / 2)
    
    merged_lines = []
    current_group = [lines[0]]

    for i, line in enumerate(lines[1:], 1):
        prev_line = current_group[-1]
        
        # Calculate vertical and horizontal distances
        prev_y = (prev_line[1] + prev_line[3]) / 2
        current_y = (line[1] + line[3]) / 2
        vertical_distance = abs(current_y - prev_y)
        
        horizontal_overlap = min(prev_line[2], line[2]) - max(prev_line[0], line[0])


        if vertical_distance < vertical_threshold and horizontal_overlap > -horizontal_threshold:
            current_group.append(line)
        else:
            x_min = min(l[0] for l in current_group)
            x_max = max(l[2] for l in current_group)
            y_avg = np.mean([l[1] for l in current_group] + [l[3] for l in current_group])
            merged_line = [x_min, y_avg, x_max, y_avg]
            merged_lines.append(merged_line)
            
            current_group = [line]

    # Merge the last group
    if current_group:
        x_min = min(l[0] for l in current_group)
        x_max = max(l[2] for l in current_group)
        y_avg = np.mean([l[1] for l in current_group] + [l[3] for l in current_group])
        merged_line = [x_min, y_avg, x_max, y_avg]
        merged_lines.append(merged_line)

    # recursively execute till no lines can be merged
    if len(merged_lines) < len(lines):
        return merge_lines(merged_lines, vertical_threshold, horizontal_threshold)
    else:
        return merged_lines


def line_in_area(line, area):
    # Check if the line lies in a certain region
    for area_line in area:
        if (abs(line[1] - area_line[1]) < 5) and (abs(line[3] - area_line[3]) < 5):
            return True
    return False


def get_column_data(image_path, center_y, horizontal_pixel_length, horizontal_actual_length):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Calculate the conversion factor from pixels to inches
    pixels_per_inch = horizontal_pixel_length / horizontal_actual_length

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eroded = cv2.erode(gray, np.ones((2, 2), np.uint8))

    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(eroded)[0]

    if lines is None:
        print("No lines detected.")
        return []

    lines = [line[0].tolist() for line in lines]
    horizontal_lines, _, _ = segment_lines(lines)
    horizontal_lines = remove_duplicate_lines(horizontal_lines)

    above_center_lines = [line for line in horizontal_lines if line[1] < center_y]
    above_center_lines.sort(key=lambda x: x[1])

    merged_above_center_lines = merge_lines(above_center_lines)

    topmost_area = []
    second_topmost_area = []

    if merged_above_center_lines:
        topmost_y = merged_above_center_lines[0][1]
        topmost_area = [l for l in merged_above_center_lines if abs(l[1] - topmost_y) < 5]

        for line in merged_above_center_lines:
            if abs(line[1] - topmost_y) >= 5:
                second_topmost_y = line[1]
                second_topmost_area = [l for l in merged_above_center_lines if abs(l[1] - second_topmost_y) < 5]
                break

        label_topmost = 'C' if second_topmost_area else 'B'

    output_image = np.copy(image)
    detection_sequence = []

    for area, color, label in [(topmost_area, (255, 0, 0), label_topmost), 
                               (second_topmost_area, (0, 255, 0), 'B')]:
        for line in area:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(output_image, (x1, y1), (x2, y2), color, 4)
            pixel_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            inch_length = pixel_length / pixels_per_inch
            detection_sequence.append((x1, label, inch_length))
            mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.putText(output_image, f"{label} {inch_length:.2f}\"", (int(mid_point[0] - 10), int(mid_point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    detection_sequence.sort()
    detection_sequence = [(label, length) for x1, label, length in detection_sequence]

    # plt.figure(figsize=(10, 10))
    cv2.imwrite("column_data.png", output_image)
    # plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    # plt.title('Detected Horizontal Lines with Lengths in Inches')
    # plt.axis('off')
    # plt.show()

    return detection_sequence

if __name__ == "__main__":
    image_path = "public/Beams/beam_1.png"
    center_y = 250  # Example y-coordinate for the center line
    try:
        detection_sequence = get_column_data(image_path, center_y)
        if detection_sequence:
            print("Detection sequence:", detection_sequence)
        else:
            print("No lines detected above the center line.")
    except ValueError as e:
        print(f"Error: {e}")
