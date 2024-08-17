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

def find_longest_horizontal_line(processed_image):
    if len(processed_image.shape) == 3:  # If the image has 3 channels, convert to grayscale
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = processed_image

    edges = cv2.Canny(gray_image, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(dilated)[0]

    if lines is not None:
        max_len = 0
        longest_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:
                length = abs(x2 - x1)
                if length > max_len:
                    max_len = length
                    longest_line = (int(x1), int(y1), int(x2), int(y2))
        return longest_line, max_len
    else:
        return None, 0

def read_text(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    return results

def detect_symbols(roi):
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=3)

    if lines is not None:
        if len(lines) == 2:
            return '"'
        elif len(lines) == 1:
            return "'"
    return ""

def get_horizontal_scale(image_path, scale_color):
    image = cv2.imread(image_path)
    processed_image, _ = preprocess_image(image, scale_color)
    longest_line, line_length = find_longest_horizontal_line(processed_image)
    
    if longest_line:
        x1, y1, x2, y2 = longest_line

        # Draw the longest horizontal line on the image
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Save the image with the longest line drawn
        output_path = os.path.splitext(image_path)[0] + '_with_line.png'
        cv2.imwrite(output_path, processed_image)

    else:
        print("No horizontal line detected.")

    # Optionally, detect and display text from the original image
    text_results = read_text(image)
    detected_texts = []

    if not text_results:
        print("Could not read any text")
    else:
        print("Detected text:")
        for result in text_results:
            top_left = tuple(result[0][0])
            bottom_right = tuple(result[0][2])
            text = result[1]
            confidence = result[2]

            expanded_top_left = (max(top_left[0] - 10, 0), max(top_left[1] - 10, 0))
            expanded_bottom_right = (min(bottom_right[0] + 10, processed_image.shape[1]), min(bottom_right[1] + 10, processed_image.shape[0]))

            roi = processed_image[expanded_top_left[1]:expanded_bottom_right[1], expanded_top_left[0]:expanded_bottom_right[0]]

            symbol = detect_symbols(roi)

            if symbol:
                if text[-1] in {'"', "'"}:
                    text = text[:-1] + symbol
                else:
                    text += symbol

            detected_texts.append((text, confidence))
            # print(f"Detected text: '{text}' with confidence: {confidence}")
            cv2.rectangle(processed_image, top_left, bottom_right, (0, 255, 0), 2)
        
        output_text_path = os.path.splitext(image_path)[0] + '_with_text.png'
        cv2.imwrite(output_text_path, processed_image)
    

    return line_length, detected_texts

if __name__ == "__main__":
    image_path = 'public/horizontal_scales/horizontal_scale_5.png'
    scale_color = "yellow"
    get_horizontal_scale(image_path, scale_color)
