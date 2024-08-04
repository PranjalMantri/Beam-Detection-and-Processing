import cv2
import easyocr
import warnings
from sklearn.cluster import DBSCAN
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

def detect_and_merge_text(image_path, output_path='output_image.jpg'):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Specify the languages you want to read

    # Load image
    image = cv2.imread(image_path)

    # Perform text detection
    results = reader.readtext(image)

    # Extract bounding box coordinates and their centers
    boxes = []
    for (bbox, text, prob) in results:
        top_left = tuple(bbox[0])
        bottom_right = tuple(bbox[2])
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2
        boxes.append((top_left, bottom_right))

    # Convert to numpy array for DBSCAN
    centers = np.array([( (box[0][0] + box[1][0]) // 2, ( (box[0][1] + box[1][1]) // 2) ) for box in boxes])

    # Perform DBSCAN clustering
    db = DBSCAN(eps=50, min_samples=1).fit(centers)
    labels = db.labels_

    # Merge boxes
    merged_boxes = []
    for label in set(labels):
        label_indices = np.where(labels == label)[0]
        if len(label_indices) > 0:
            x_min = min(boxes[i][0][0] for i in label_indices)
            y_min = min(boxes[i][0][1] for i in label_indices)
            x_max = max(boxes[i][1][0] for i in label_indices)
            y_max = max(boxes[i][1][1] for i in label_indices)
            
            merged_boxes.append(((x_min, y_min), (x_max, y_max)))

    # Remove small rectangles inside larger ones
    final_boxes = []
    for i in range(len(merged_boxes)):
        is_inside = False
        for j in range(len(merged_boxes)):
            if i != j:
                if (merged_boxes[i][0][0] >= merged_boxes[j][0][0] and
                    merged_boxes[i][0][1] >= merged_boxes[j][0][1] and
                    merged_boxes[i][1][0] <= merged_boxes[j][1][0] and
                    merged_boxes[i][1][1] <= merged_boxes[j][1][1]):
                    is_inside = True
                    break
        if not is_inside:
            final_boxes.append(merged_boxes[i])

    detected_texts = []

    # Draw final rectangles around detected text and run OCR again on merged areas
    for (top_left, bottom_right) in final_boxes:
        # Extract the merged region from the image
        merged_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        # Run OCR again on the merged region
        new_results = reader.readtext(merged_region)
        
        # Merge the detected text
        merged_text = " ".join([text for (bbox, text, prob) in new_results if text.strip()])
        
        # Only draw and store rectangles with detected text
        if merged_text:
            # Draw rectangle
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            
            # Put merged text near the bounding box
            cv2.putText(image, merged_text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print rectangle coordinates and detected text
            print(f"Rectangle: Top Left {top_left}, Bottom Right {bottom_right}")
            print(f"Detected Text: {merged_text}\n")

            # Append the detected text and coordinates to the list
            detected_texts.append({
                'coordinates': (top_left, bottom_right),
                'text': merged_text
            })

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Output image saved to {output_path}")

    return detected_texts

# Example usage
image_path = "public/Beams/beam_0.png"
output_path = 'output_image.jpg'
detected_texts = detect_and_merge_text(image_path, output_path)
