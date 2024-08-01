import cv2
import numpy as np

def clean_mask_image(image_path, type="beam"):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if type == "beam":
        # Define range for red color in HSV
        lower_color1 = np.array([0, 100, 100])
        upper_color1 = np.array([10, 255, 255])
        lower_color2 = np.array([160, 100, 100])
        upper_color2 = np.array([180, 255, 255])
        color = (0, 0, 255)  # Red color in BGR
        output_path = "public/beam-image/cleaned_masked_image.png"
    elif type == "column":
        # Define range for cyan color in HSV
        lower_color1 = np.array([80, 100, 100])
        upper_color1 = np.array([90, 255, 255])
        lower_color2 = np.array([90, 100, 100])
        upper_color2 = np.array([100, 255, 255])
        color = (255, 255, 0)  # Cyan color in BGR
        output_path = "public/column-image/cleaned_masked_image.png"
    else:
        raise ValueError("Invalid type. Please use 'beam' or 'column'.")
    
    # Create masks for the selected color
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
    color_mask = mask1 + mask2
    
    # Reconnect broken horizontal lines
    kernel_close = np.ones((1, 15), np.uint8)  # Horizontal kernel
    color_lines = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Use horizontal structuring element to keep only horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(color_lines, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Find contours of the horizontal lines
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the reconnected horizontal lines
    result = img.copy()
    cv2.drawContours(result, contours, -1, color, 1)
    
    # Save the result image
    cv2.imwrite(output_path, result)
    
    return output_path
