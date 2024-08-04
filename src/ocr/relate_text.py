import easyocr
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load image
image_path = "public/Beams/beam_0.png"
image = cv2.imread(image_path)

if image is None:
    print("Something went wrong while reading the image")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Canny edge detector
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use Hough Line Transform to detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest rectangle contour is the beam
beam_contour = max(contours, key=cv2.contourArea)

# Draw the beam contour for visualization
beam_contour_image = image.copy()
cv2.drawContours(beam_contour_image, [beam_contour], -1, (255, 0, 0), 2)

# Initialize easyOCR reader
reader = easyocr.Reader(['en'])

# Detect text
results = reader.readtext(image)

# Create a list to store filtered results
filtered_results = []

# Define a function to check if a point is near a line
def is_point_near_line(point, line, threshold=10):
    x1, y1, x2, y2 = line
    px, py = point
    distance = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return distance < threshold

# Filter text near lines and outside the beam
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    text_center = (int((top_left[0] + bottom_right[0]) // 2), int((top_left[1] + bottom_right[1]) // 2))
    
    # Check if text center is inside the beam contour
    inside_beam = cv2.pointPolygonTest(beam_contour, text_center, False) >= 0
    
    # Check if the text is near any detected lines
    near_line = any(is_point_near_line(text_center, line[0]) for line in lines)
    
    if near_line and not inside_beam:
        filtered_results.append((bbox, text, prob))

# Draw rectangles around filtered text
for (bbox, text, prob) in filtered_results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Draw the rectangular beam for visualization
cv2.drawContours(image, [beam_contour], -1, (255, 0, 0), 2)

# Display the image with rectangles and the beam contour
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print the text and its bounding box coordinates
for (bbox, text, prob) in filtered_results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    print(f"Text: {text}, Coordinates: top_left={top_left}, bottom_right={bottom_right}")
