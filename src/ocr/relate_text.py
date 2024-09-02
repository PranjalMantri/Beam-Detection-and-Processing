import cv2
import numpy as np

def calculate_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def get_center(bbox):
    center_x = (bbox[0][0] + bbox[1][0]) / 2
    center_y = (bbox[0][1] + bbox[1][1]) / 2
    return (center_x, center_y)
    

def get_relations(image_path, bars, texts, output_path='output_image_relations.jpg'):
    # Load the image
    image = cv2.imread(image_path)

    # Extract centers of bars and texts
    bar_centers = []
    for bar in bars:
        x1, y1, x2, y2 = bar['horizontal_bar']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bar_centers.append({'center': (center_x, center_y), 'type': bar['type'], 'bar': bar, 'coordinates': [(int(x1), int(y1)), (int(x2), int(y2))]})

    text_centers = [{'center': get_center(text['coordinates']), 'text': text['text'], 'coordinates': [(int(text['coordinates'][0][0]), int(text['coordinates'][0][1])), (int(text['coordinates'][1][0]), int(text['coordinates'][1][1]))]} for text in texts]

    # Associate text with bars considering vertical position
    associations = []
    for bar in bar_centers:
        bar_center = bar['center']
        bar_type = bar['type']
        closest_text = None
        min_distance = float('inf')
        for text in text_centers:
            text_center = text['center']
            if (bar_type == 'Top Steel' and text_center[1] < bar_center[1]) or (bar_type == 'Bottom Steel' and text_center[1] > bar_center[1]):
                distance = calculate_distance(bar_center, text_center)
                if distance < min_distance:
                    min_distance = distance
                    closest_text = text
        
        if closest_text:
            associations.append((bar, closest_text))

    # Visualize the bars, text, and their associations
    for bar, text in associations:
        # Draw bar bounding box
        bar_start, bar_end = bar['coordinates']
        cv2.rectangle(image, bar_start, bar_end, (0, 255, 0), 2)

        # Draw text bounding box
        text_start, text_end = text['coordinates']
        cv2.rectangle(image, text_start, text_end, (255, 0, 0), 2)

        # Draw line connecting bar and text
        cv2.line(image, (int(bar['center'][0]), int(bar['center'][1])), (int(text['center'][0]), int(text['center'][1])), (0, 0, 255), 2)

    # Save or display the image
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Example usage
    image_path = "public/beam-image/cleaned_masked_image.png"
    bars = [{'horizontal_bar': [61.874992, 270.12292, 939.375, 270.68887], 'start_vertical_bars': [[59.39831, 139.37581, 59.313747, 273.12656]], 'end_vertical_bars': [[939.0522, 141.87325, 939.1515, 265.6262]], 'horizontal_length': 877.5002, 'vertical_length': 257.5037612915039, 'total_length': 1135.0039443969727, 'type': 'Bottom Steel'}, {'horizontal_bar': [56.874996, 126.98653, 943.125, 126.98558], 'start_vertical_bars': [[51.92357, 128.12518, 51.904232, 260.62518]], 'end_vertical_bars': [[945.06525, 131.87604, 945.2979, 256.8756]], 'horizontal_length': 886.25, 'vertical_length': 257.4997863769531, 'total_length': 1143.7497863769531, 'type': 'Top Steel'}, {'horizontal_bar': [213.08562, 267.23352, 796.8751, 267.59586], 'start_vertical_bars': [], 'end_vertical_bars': [], 'horizontal_length': 583.7896, 'vertical_length': 0, 'total_length': 583.7896118164062, 'type': 'Bottom Steel'}]

    texts = [{'coordinates': ((430, 34), (536, 66)), 'text': '2-20 &A)'}, {'coordinates': ((296, 92), (342, 122)), 'text': '105'}, {'coordinates': ((609, 187), (895, 223)), 'text': '2-10 €XTRA RING TYP.'}, {'coordinates': ((296, 292), (342, 324)), 'text': '195]'}, {'coordinates': ((429, 339), (765, 377)), 'text': '2-20 (B)+2-25 )+2-25'}, {'coordinates': ((140, 434), (230, 522)), 'text': '8q 4"CIC 12NOS'}, {'coordinates': ((403, 437), (595, 559)), 'text': '8q 6"CIC REST B105 12"x24")'}, {'coordinates': ((800, 431), (890, 522)), 'text': '4"CIC 12NOS 8Q'}]

    get_relations(image_path, bars, texts)
