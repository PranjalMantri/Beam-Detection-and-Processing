import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from src.data.helper.line_processing import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def merge_lines(lines, vertical_threshold=5, horizontal_threshold=20):
    if not lines:
        return []

    logger.info(f"Starting merge_lines with {len(lines)} lines")
    logger.info(f"Vertical threshold: {vertical_threshold}, Horizontal threshold: {horizontal_threshold}")

    # Sort lines by y-coordinate
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

        logger.info(f"Comparing line {i-1} and line {i}:")
        logger.info(f"  Vertical distance: {vertical_distance:.2f}")
        logger.info(f"  Horizontal overlap: {horizontal_overlap:.2f}")

        if vertical_distance < vertical_threshold and horizontal_overlap > -horizontal_threshold:
            current_group.append(line)
            logger.info("  Lines merged into current group")
        else:
            # Merge the current group
            x_min = min(l[0] for l in current_group)
            x_max = max(l[2] for l in current_group)
            y_avg = np.mean([l[1] for l in current_group] + [l[3] for l in current_group])
            merged_line = [x_min, y_avg, x_max, y_avg]
            merged_lines.append(merged_line)
            
            logger.info(f"  Merging group of {len(current_group)} lines:")
            logger.info(f"    Merged line: {merged_line}")
            
            # Start a new group
            current_group = [line]

    # Merge the last group
    if current_group:
        x_min = min(l[0] for l in current_group)
        x_max = max(l[2] for l in current_group)
        y_avg = np.mean([l[1] for l in current_group] + [l[3] for l in current_group])
        merged_line = [x_min, y_avg, x_max, y_avg]
        merged_lines.append(merged_line)
        
        logger.info(f"Merging final group of {len(current_group)} lines:")
        logger.info(f"  Merged line: {merged_line}")

    logger.info(f"Merge complete. {len(merged_lines)} merged lines produced.")

    # Recursive call if the number of lines has changed
    if len(merged_lines) < len(lines):
        logger.info("Number of lines reduced. Performing another merge pass.")
        return merge_lines(merged_lines, vertical_threshold, horizontal_threshold)
    else:
        logger.info("No further merging possible. Returning final result.")
        return merged_lines


def line_in_area(line, area):
    for area_line in area:
        if (abs(line[1] - area_line[1]) < 5) and (abs(line[3] - area_line[3]) < 5):
            return True
    return False


def get_column_data(image_path, center_y):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or unable to load.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eroded = cv2.erode(gray, np.ones((2, 2), np.uint8))
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(eroded)[0]

    if lines is None:
        print("No lines detected.")
        return []

    lines = [line[0].tolist() for line in lines]
    horizontal_lines, vertical_lines, slanted_lines = segment_lines(lines)
    horizontal_lines = remove_duplicate_lines(horizontal_lines)

    above_center_lines = [line for line in horizontal_lines if line[1] < center_y]
    above_center_lines.sort(key=lambda x: x[1])

    # Merge lines before categorizing them
    merged_above_center_lines = merge_lines(above_center_lines)
    merged_above_center_lines = merge_lines(merged_above_center_lines)

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

        if not second_topmost_area:
            label_topmost = 'B'
        else:
            label_topmost = 'C'

    # Print the number of lines in each section
    print(f"Number of lines in topmost area: {len(topmost_area)}")
    print(f"Number of lines in second topmost area: {len(second_topmost_area)}")

    output_image = np.copy(image)
    detection_sequence = []

    # Draw only the lines in topmost_area and second_topmost_area
    for area, color, label in [(topmost_area, (255, 0, 0), label_topmost), 
                               (second_topmost_area, (0, 255, 0), 'B')]:
        for line in area:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(output_image, (x1, y1), (x2, y2), color, 4)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            detection_sequence.append((x1, label, length))
            mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.putText(output_image, f"{label} {length:.1f}", (int(mid_point[0] - 10), int(mid_point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    detection_sequence.sort()  # Sort based on the x-coordinate
    detection_sequence = [(label, length) for x1, label, length in detection_sequence]

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Horizontal Lines with Highlighted Center Division')
    plt.axis('off')
    plt.show()

    return detection_sequence

# Example usage
if __name__ == "__main__":
    image_path = "public/Beams/beam_1.png"
    center_y = 250  # Example y-coordinate for the center line
    try:
        detection_sequence = get_column_data(image_path=image_path, center_y=center_y)
        if detection_sequence:
            print("Detection sequence:", detection_sequence)
        else:
            print("No lines detected above the center line.")
    except ValueError as e:
        print(f"Error: {e}")
