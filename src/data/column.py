import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.data.helper.line_processing import *

def get_column_data(image_path):
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

    horizontal_lines.sort(key=lambda x: x[1])
    topmost_y = horizontal_lines[0][1]
    topmost_area = [line for line in horizontal_lines if abs(line[1] - topmost_y) < 5]

    second_topmost_y = None
    for line in horizontal_lines:
        if abs(line[1] - topmost_y) >= 5:
            second_topmost_y = line[1]
            break
    second_topmost_area = [line for line in horizontal_lines if second_topmost_y and abs(line[1] - second_topmost_y) < 5]

    topmost_length = sum(np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2) for line in topmost_area)
    second_topmost_length = sum(np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2) for line in second_topmost_area)

    # print(f"Number of lines in the topmost area (C): {len(topmost_area)}, Length: {topmost_length}")
    # print(f"Number of lines in the second topmost area (B): {len(second_topmost_area)}, Length: {second_topmost_length}")

    output_image = np.copy(image)
    detection_sequence = []

    for line in horizontal_lines:
        x1, y1, x2, y2 = map(int, line)
        color = (0, 0, 255)
        label = None

        if line in topmost_area:
            color = (0, 255, 0)
            label = "C"
        elif line in second_topmost_area:
            color = (255, 0, 0)
            label = "B"

        if label:
            length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
            detection_sequence.append((x1, label, length))
            mid_point = ((line[0] + line[2]) // 2, (line[1] + line[3]) // 2)
            cv2.putText(output_image, f"{label} {length:.1f}", (int(mid_point[0] - 10), int(mid_point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.line(output_image, (x1, y1), (x2, y2), color, 2)

    detection_sequence.sort()  # Sort based on the x-coordinate
    detection_sequence = [(label, length) for x1, label, length in detection_sequence]

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Horizontal Lines')
    plt.axis('off')
    plt.show()

    return detection_sequence

if __name__ == "__main__":
        # Use the function
    image_path = "public/column-image/cleaned_masked_image.png"
    detection_sequence = get_column_data(image_path)
    print(detection_sequence)
