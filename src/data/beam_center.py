import cv2
import numpy as np

def get_center_height(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create LSD detector
    lsd = cv2.createLineSegmentDetector(0)
    
    # Detect lines in the image
    lines = lsd.detect(gray)[0]  # Position 0 of the returned tuple are the detected lines

    # Filter horizontal lines with length at least 50 pixels
    horizontal_lines = [
        line for line in lines
        if abs(line[0][1] - line[0][3]) < 10 and np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2) >= 100
    ]
    
    if not horizontal_lines:
        print("No horizontal lines detected")
        return None
    
    # Find topmost and bottommost lines
    topmost_line = min(horizontal_lines, key=lambda line: line[0][1])
    bottommost_line = max(horizontal_lines, key=lambda line: line[0][1])
    
    # Calculate center line
    center_y = int((topmost_line[0][1] + bottommost_line[0][1]) / 2)
    
    return center_y

# Example usage
if __name__ == "__main__":
    image_path = "public/Beams/beam_9.png"
    try:
        center_y = get_center_height(image_path=image_path)
        if center_y is not None:
            print(f"Center line height coordinate: {center_y}")
        else:
            print("No horizontal lines detected")
    except ValueError as e:
        print(f"Error: {e}")
