import cv2
import numpy as np
import easyocr
import warnings
import os

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define HSV color ranges for different colors
COLOR_RANGES = {
    "red": ((0, 50, 50), (10, 255, 255)),
    "green": ((40, 50, 50), (80, 255, 255)),
    "blue": ((100, 50, 50), (140, 255, 255)),
    "yellow": ((20, 50, 50), (30, 255, 255)),
    "cyan": ((80, 50, 50), (100, 255, 255)),
    "magenta": ((140, 50, 50), (170, 255, 255)),
    "orange": ((10, 50, 50), (20, 255, 255)),
    "purple": ((130, 50, 50), (160, 255, 255)),
    "black": ((0, 0, 0), (180, 255, 50)),
}

def preprocess_image(image, color_name):
    """
    Preprocess the image by filtering out non-color areas and converting to grayscale.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        color_name (str): Name of the color to filter for.

    Returns:
        tuple: Grayscale image with color areas filtered out and color-filtered result.
    """
    if color_name.lower() not in COLOR_RANGES:
        raise ValueError(f"Unsupported color name '{color_name}'. Supported colors: {list(COLOR_RANGES.keys())}")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound, upper_bound = COLOR_RANGES[color_name.lower()]
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
    grayscale_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    return grayscale_result, result

def find_longest_horizontal_line(processed_image, text_bounding_boxes):
    """
    Find the longest horizontal line below text bounding boxes.

    Args:
        processed_image (numpy.ndarray): Grayscale image with filtered color.
        text_bounding_boxes (list): List of bounding boxes around text.

    Returns:
        tuple: Coordinates of the longest horizontal line and its length.
    """
    # Get the image dimensions
    height, width = processed_image.shape[:2]

    # Determine the minimum y-coordinate below which to search for lines
    min_y = max([box[2][1] for box in text_bounding_boxes], default=0)
    
    # Perform edge detection
    edges = cv2.Canny(processed_image, 50, 150)

    # Use HoughLinesP for line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=width//4, maxLineGap=20)

    # Copy of the original image to draw lines on
    lines_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    max_len = 0
    longest_line = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if the line is below the text
            if y1 > min_y and y2 > min_y:
                # Ensure the line is approximately horizontal
                if abs(y2 - y1) < 5:
                    length = abs(x2 - x1)
                    if length > max_len:
                        max_len = length
                        longest_line = (int(x1), int(y1), int(x2), int(y2))

            # Draw all detected lines on the image
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the longest line in red for clarity
    if longest_line:
        cv2.line(lines_image, (longest_line[0], longest_line[1]), 
                 (longest_line[2], longest_line[3]), (0, 0, 255), 3)


    return longest_line, max_len

def read_text(image):
    """
    Read text from an image using EasyOCR.

    Args:
        image (numpy.ndarray): Image from which to read text.

    Returns:
        list: List of text detection results.
    """
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    return results

def detect_symbols(roi):
    """
    Detect symbols (quotes) in a region of interest (ROI).

    Args:
        roi (numpy.ndarray): Region of interest image.

    Returns:
        str: Detected symbol, either '"' or "'".
    """
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=3)

    if lines is not None:
        if len(lines) == 2:
            return '"'
        elif len(lines) == 1:
            return "'"
    return ""

def get_horizontal_scale(image_path, scale_color):
    """
    Process an image to find and annotate the longest horizontal line and text information.

    Args:
        image_path (str): Path to the image file.
        scale_color (str): Color of the horizontal scale in the image.

    Returns:
        tuple: Length of the longest line and detected text information.
    """
    image_path = f"public/horizontal_scales/horizontal_scale_{image_path}.png"
    image = cv2.imread(image_path)

    # Preprocess the image
    processed_image, _ = preprocess_image(image, scale_color)

    # Read text from the processed image
    text_results = read_text(processed_image)

    detected_texts = []
    text_bounding_boxes = []

    if text_results:
        for result in text_results:
            top_left = tuple(result[0][0])
            bottom_right = tuple(result[0][2])
            text = result[1]
            confidence = result[2]

            # Expand the bounding box around the detected text
            expanded_top_left = (max(top_left[0] - 10, 0), max(top_left[1] - 10, 0))
            expanded_bottom_right = (min(bottom_right[0] + 10, processed_image.shape[1]), min(bottom_right[1] + 10, processed_image.shape[0]))

            # Extract region of interest (ROI) for symbol detection
            roi = processed_image[expanded_top_left[1]:expanded_bottom_right[1], expanded_top_left[0]:expanded_bottom_right[0]]

            # Detect symbols in the ROI
            symbol = detect_symbols(roi)

            if symbol:
                if text[-1] in {'"', "'"}:
                    text = text[:-1] + symbol
                else:
                    text += symbol

            detected_texts.append((text, confidence))
            text_bounding_boxes.append((top_left, bottom_right, expanded_bottom_right))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        
        # Save the image with annotated text
        output_text_path = os.path.splitext(image_path)[0] + '_with_text.png'
        cv2.imwrite(output_text_path, image)

    # Find the longest horizontal line below the text
    longest_line, line_length = find_longest_horizontal_line(processed_image, text_bounding_boxes)
    
    if longest_line:
        x1, y1, x2, y2 = longest_line

        # Draw the longest line on the image
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Annotate the image with the length of the line
        text = f"Length: {line_length} px"
        text_position = (x1, y1 - 10) if y1 - 10 > 10 else (x1, y1 + 20)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Save the image with the longest line annotated
        output_path = os.path.splitext(image_path)[0] + '_with_line.png'
        cv2.imwrite(output_path, image)
    else:
        print("No horizontal line detected.")

    return line_length, detected_texts

if __name__ == "__main__":
    # Define the path to the image and color used for scale detection
    image_path = '5'  # Example: use '5' for the filename 'horizontal_scale_5.png'
    scale_color = "yellow"
    get_horizontal_scale(image_path, scale_color)
