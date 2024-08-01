import cv2
import numpy as np
import easyocr
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)


def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Apply edge detection
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Dilate the edges to connect broken lines
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    return dilated, image


def find_longest_vertical_line(processed_image):
    # Create LSD detector
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(processed_image)[0]  # Detect lines in the image

    if lines is not None:
        # Find the longest vertical line
        max_len = 0
        longest_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 5:  # Allow small deviation from vertical
                length = abs(y2 - y1)
                if length > max_len:
                    max_len = length
                    longest_line = (int(x1), int(y1), int(x2), int(y2))
        
        return longest_line
    else:
        return None


def read_text(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    return results


def detect_symbols(roi):
    # Use edge detection and Hough Line Transform to detect lines
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=3)
    
    if lines is not None:
        if len(lines) == 2:
            return '"'
        elif len(lines) == 1:
            return "'"
    return ""


def get_vertical_scale(image_path="public/vertical_scales/vertical_scale_0.png"):
    processed_image, original_image = preprocess_image(image_path)
    longest_line = find_longest_vertical_line(processed_image)
    
    # Convert grayscale to color for drawing colored line
    color_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    line_length = None
    if longest_line:
        x1, y1, x2, y2 = longest_line
        line_length = abs(y2 - y1)
        
        # Draw the line on the color image
        cv2.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red line
        
        # Save the image with the line drawn
        output_path = os.path.splitext(image_path)[0] + '_with_line.png'
        cv2.imwrite(output_path, color_image)
        
        # Display the image
        # cv2.imshow('Longest Vertical Line', color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # else:
    #     print("No vertical line detected.")
    
    rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
    
    text_results = read_text(rotated_image)

    detected_texts = []

    # if not text_results:
    #     print("Could not read any text")

    for result in text_results:
        top_left = tuple(result[0][0])
        bottom_right = tuple(result[0][2])
        text = result[1]
        confidence = result[2]
        
        # Expand the bounding box
        expanded_top_left = (max(top_left[0] - 10, 0), max(top_left[1] - 10, 0))
        expanded_bottom_right = (min(bottom_right[0] + 10, rotated_image.shape[1]), min(bottom_right[1] + 10, rotated_image.shape[0]))
        
        # Extract the ROI
        roi = rotated_image[expanded_top_left[1]:expanded_bottom_right[1], expanded_top_left[0]:expanded_bottom_right[0]]
        
        # Detect symbols
        symbol = detect_symbols(roi)

        if symbol:
            if text[-1] in {'"', "'"}:
                text = text[:-1] + symbol
            else:
                text += symbol
        
        
        detected_texts.append((text, confidence))
        
        cv2.rectangle(rotated_image, top_left, bottom_right, (0, 255, 0), 2)
    
    # Save and display the image with rectangles
    output_text_path = os.path.splitext(image_path)[0] + '_with_text.png'
    cv2.imwrite(output_text_path, rotated_image)

    return line_length, detected_texts



if __name__ == "__main__":
    image_path = 'public/vertical_scales/vertical_scale_0.png'  # Replace with your image path
    get_vertical_scale(image_path)
