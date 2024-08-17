import cv2
import numpy as np
import easyocr
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

COLOR_RANGES = {
    "red": ((0, 50, 50), (10, 255, 255)),  # Red hue range
    "green": ((40, 50, 50), (80, 255, 255)),  # Green hue range
    "blue": ((100, 50, 50), (140, 255, 255)),  # Blue hue range
    "yellow": ((20, 50, 50), (30, 255, 255)),  # Yellow hue range
    "cyan": ((80, 50, 50), (100, 255, 255)),  # Cyan hue range
    "magenta": ((140, 50, 50), (170, 255, 255)),  # Magenta hue range
    "orange": ((10, 50, 50), (20, 255, 255)),  # Orange hue range
    "purple": ((130, 50, 50), (160, 255, 255)),  # Purple hue range
}

def preprocess_image(image, color_name):
    if color_name.lower() not in COLOR_RANGES:
        raise ValueError(f"Unsupported color name '{color_name}'. Supported colors: {list(COLOR_RANGES.keys())}")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound, upper_bound = COLOR_RANGES[color_name.lower()]
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
    grayscale_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    return grayscale_result, result

def find_longest_vertical_line(processed_image, original_image, line_color):
    processed_image, _ = preprocess_image(original_image, line_color)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(processed_image)[0]

    if lines is None:
        return None

    debug_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    max_len = 0
    longest_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Access the first (and only) element of the line array
        if abs(x2 - x1) < 5:
            length = abs(y2 - y1)
            if length > max_len:
                max_len = length
                longest_line = (int(x1), int(y1), int(x2), int(y2))

        # Draw all detected lines
        cv2.line(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    if longest_line:
        x1, y1, x2, y2 = longest_line
        cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    debug_output_path = "detected_lines.png"
    cv2.imwrite(debug_output_path, debug_image)

    return longest_line

def detect_symbols(roi):
    # Use edge detection
    edges = cv2.Canny(roi, 50, 150)
    
    # Use Hough Line Transform to detect line
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=5, maxLineGap=3)

    if lines is not None:
        if len(lines) == 2:
            return '"'
        elif len(lines) == 1:
            return "'"
    return ""


def get_vertical_scale(image_path, scale_color):
    image_path = f"public/vertical_scales/{image_path}.png"
    original_image = cv2.imread(image_path)

    longest_line = find_longest_vertical_line(original_image, original_image, scale_color)
    
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.rotate(grayscale_image, cv2.ROTATE_90_CLOCKWISE)
    
    line_length = None
    if longest_line:
        x1, y1, x2, y2 = longest_line
        line_length = abs(y2 - y1)
        
        cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        output_path = os.path.splitext(image_path)[0] + '_with_line.png'
        cv2.imwrite(output_path, original_image)
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Perform OCR on the image
    results = reader.readtext(grayscale_image)
    
    detected_texts = []

    for result in results:
        top_left = tuple(result[0][0])
        bottom_right = tuple(result[0][2])
        text = result[1]
        confidence = result[2]
        
        # Expand the bounding box
        expanded_top_left = (int(max(top_left[0] - 10, 0)), int(max(top_left[1] - 10, 0)))
        expanded_bottom_right = (int(min(bottom_right[0] + 10, grayscale_image.shape[1])), min(bottom_right[1] + 10, grayscale_image.shape[0]))
        
        # Extract the ROI
        roi = grayscale_image[expanded_top_left[1]:expanded_bottom_right[1], expanded_top_left[0]:expanded_bottom_right[0]]

        # Detect symbols
        symbol = detect_symbols(roi)

        if symbol:
            if text[-1] in {'"', "'"}:
                text = text[:-1] + symbol
            else:
                text += symbol
        
        detected_texts.append((text, confidence))
        
        cv2.rectangle(grayscale_image, top_left, bottom_right, (0, 255, 0), 2)
    
    # Save and display the image with rectangles
    output_text_path = os.path.splitext(image_path)[0] + '_with_text.png'
    cv2.imwrite(output_text_path, grayscale_image)

    # print(f"The length of the longest line is: {line_length}")
    # print(f"Detected Texts: {detected_texts}")
    return line_length, detected_texts


if __name__ == "__main__":
    image_path = 'public/vertical_scales/vertical_scale_8.png'
    scale_color = "blue"
    get_vertical_scale(image_path, scale_color)
