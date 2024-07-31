import cv2
import numpy as np
import easyocr
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Create a mask for non-gray pixels
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([180, 50, 255])  # Adjust the upper bound to define what is considered "gray"
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)  # Invert the mask to get non-gray areas
    
    # Apply the mask to the original image
    image[mask > 0] = [255, 255, 255]  # Convert non-gray areas to white
    
    # Display the image for debugging purposes
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return image

def find_longest_horizontal_line(processed_image):
    # Convert image to grayscale for line detection
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Dilate the edges to connect broken lines
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Create LSD detector
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(dilated)[0]  # Detect lines in the image

    if lines is not None:
        # Find the longest horizontal line
        max_len = 0
        longest_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # Allow small deviation from horizontal
                length = abs(x2 - x1)
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

def get_horizontal_scale(image_path):
    processed_image = preprocess_image(image_path)
    longest_line = find_longest_horizontal_line(processed_image)
    
    # Convert image to grayscale for text detection
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    line_length = None
    if longest_line:
        x1, y1, x2, y2 = longest_line
        line_length = abs(x2 - x1)
        print(f"The longest horizontal line is from ({x1}, {y1}) to ({x2}, {y2}) with length {line_length} pixels.")
        
        # Draw the line on the processed image
        cv2.line(processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red line
        
        # Save the image with the line drawn
        output_path = os.path.splitext(image_path)[0] + '_with_line.png'
        cv2.imwrite(output_path, processed_image)
        print(f"Image with detected line saved as {output_path}")
        
        # Display the image
        cv2.imshow('Longest Horizontal Line', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No horizontal line detected.")
    
    text_results = read_text(gray_image)

    detected_texts = []
    if not text_results:
        print("Could not read any text")

    for result in text_results:
        top_left = tuple(result[0][0])
        bottom_right = tuple(result[0][2])
        text = result[1]
        confidence = result[2]
        print(f"Detected text: {text} with confidence {confidence}")
        
        # Expand the bounding box
        expanded_top_left = (max(top_left[0] - 10, 0), max(top_left[1] - 10, 0))
        expanded_bottom_right = (min(bottom_right[0] + 10, gray_image.shape[1]), min(bottom_right[1] + 10, gray_image.shape[0]))
        
        # Extract the ROI
        roi = gray_image[expanded_top_left[1]:expanded_bottom_right[1], expanded_top_left[0]:expanded_bottom_right[0]]
        
        # Detect symbols
        symbol = detect_symbols(roi)
        
        # Include the detected symbol in the text
        if symbol:
            text += symbol
        
        detected_texts.append((text, confidence))
        print(f"Detected text with symbols: {text} with confidence {confidence}")
        
        cv2.rectangle(processed_image, top_left, bottom_right, (0, 255, 0), 2)
    
    # Save and display the image with rectangles
    output_text_path = os.path.splitext(image_path)[0] + '_with_text.png'
    cv2.imwrite(output_text_path, processed_image)
    print(f"Image with detected text saved as {output_text_path}")
    
    cv2.imshow('Detected Text', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return line_length, detected_texts


if __name__ == "__main__":
    image_path = 'public/horizontal_scales/horizontal_scale_2.png'  # Replace with your image path
    get_horizontal_scale(image_path)
