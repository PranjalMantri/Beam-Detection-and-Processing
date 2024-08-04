import cv2
import numpy as np
import os


def calculate_line_length(line):
    x1, y1, x2, y2 = line
    start_point = np.array((x1, y1))
    end_point = np.array((x2, y2))
    distance = np.linalg.norm(start_point - end_point)
    return distance


def sort_lines(lines):
    return sorted(lines, key=lambda x: calculate_line_length(x), reverse=True)


def calculate_distance(xa, ya, xb, yb):
    return np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)


def check_vertical_at_endpoints(horizontal_line, vertical_lines, tolerance=7, distance_tolerance=10):
    x1, y1, x2, y2 = horizontal_line
    start_lines = []
    end_lines = []

    for v_line in vertical_lines:
        vx1, vy1, vx2, vy2 = v_line
        if (abs(x1 - vx1) <= tolerance or abs(x2 - vx1) <= tolerance or abs(x1 - vx2) <= tolerance or abs(x2 - vx2) <= tolerance):
            dist1 = calculate_distance(x1, y1, vx1, vy1)
            dist2 = calculate_distance(x1, y1, vx2, vy2)
            dist3 = calculate_distance(x2, y2, vx1, vy1)
            dist4 = calculate_distance(x2, y2, vx2, vy2)

            if dist1 <= distance_tolerance or dist2 <= distance_tolerance:
                start_lines.append(v_line)
            if dist3 <= distance_tolerance or dist4 <= distance_tolerance:
                end_lines.append(v_line)
    
    return start_lines, end_lines


def segment_lines(lines, max_diff=5):
    horizontal_lines, vertical_lines, slanted_lines = [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) <= max_diff:
            vertical_lines.append([x1, y1, x2, y2])
        elif abs(y1 - y2) <= max_diff:
            horizontal_lines.append([x1, y1, x2, y2])
        else:
            slanted_lines.append([x1, y1, x2, y2])
    return horizontal_lines, vertical_lines, slanted_lines


def remove_duplicate_lines(lines, max_distance=20, max_y_diff=10, is_horizontal=True):
    def merge_lines(lines, merged_lines):
        for line in lines:
            x1, y1, x2, y2 = line
            merged = False
            for i, merged_line in enumerate(merged_lines):
                x3, y3, x4, y4 = merged_line
                if is_horizontal:
                    if ((abs(x1 - x4) < max_distance and abs(y1 - y4) < max_y_diff) or
                        (abs(x2 - x3) < max_distance and abs(y2 - y3) < max_y_diff)):
                        merged_lines[i] = [min(x1, x3), y1, max(x2, x4), y2]
                        merged = True
                        break
                else:
                    if ((abs(x1 - x3) < max_y_diff and abs(y1 - y3) < max_distance) or
                        (abs(x2 - x4) < max_y_diff and abs(y2 - y4) < max_distance)):
                        merged_lines[i] = [x1, min(y1, y3), x2, max(y2, y4)]
                        merged = True
                        break
            if not merged:
                merged_lines.append([x1, y1, x2, y2])
        return merged_lines

    merged_lines = merge_lines(lines, [])
    if len(lines) == len(merged_lines):
        return merged_lines
    else:
        return remove_duplicate_lines(merged_lines, max_distance, max_y_diff, is_horizontal)


def detect_bars(horizontal_lines, vertical_lines, image_shape):
    bar_info = []
    
    for h_line in horizontal_lines:
        h_x1, h_y1, h_x2, h_y2 = h_line
        start_v_lines, end_v_lines = check_vertical_at_endpoints(h_line, vertical_lines)
        
        start_v_lines, end_v_lines = filter_closest_lines(start_v_lines, end_v_lines)
        
        horizontal_length = calculate_line_length(h_line)
        
        if horizontal_length <= 50:  # Skip bars with horizontal length less than or equal to 50 pixels
            continue
        
        vertical_length = 0
        
        for v_line in start_v_lines + end_v_lines:
            vertical_length += calculate_line_length(v_line)
        
        total_length = horizontal_length + vertical_length
        
        if total_length > 100:
            bar_info.append({
                "horizontal_bar": [h_x1, h_y1, h_x2, h_y2],
                "start_vertical_bars": start_v_lines,
                "end_vertical_bars": end_v_lines,
                "horizontal_length": horizontal_length,
                "vertical_length": vertical_length,
                "total_length": total_length,
            })
        
    return bar_info


def filter_closest_lines(start_lines, end_lines):
    if start_lines:
        start_lines = sorted(start_lines, key=lambda x: x[1])[:1]
    if end_lines:
        end_lines = sorted(end_lines, key=lambda x: x[1])[:1]
        
    return start_lines, end_lines


def draw_bar(image, bar_info, bar_number, h_pixels_to_inches, v_pixels_to_inches):
    if bar_info["total_length"] > 150:
        bar_image = image.copy()
        
        cv2.line(bar_image, 
                 (int(bar_info["horizontal_bar"][0]), int(bar_info["horizontal_bar"][1])),
                 (int(bar_info["horizontal_bar"][2]), int(bar_info["horizontal_bar"][3])), 
                 (0, 255, 0), 2)
        
        for v_bar in bar_info["start_vertical_bars"]:
            cv2.line(bar_image, 
                     (int(v_bar[0]), int(v_bar[1])), 
                     (int(v_bar[2]), int(v_bar[3])), 
                     (0, 255, 0), 2)
        
        for v_bar in bar_info["end_vertical_bars"]:
            cv2.line(bar_image, 
                     (int(v_bar[0]), int(v_bar[1])), 
                     (int(v_bar[2]), int(v_bar[3])), 
                     (0, 255, 0), 2)
        
        horizontal_length_inch = bar_info['horizontal_length'] * h_pixels_to_inches
        vertical_length_inch = bar_info['vertical_length'] * v_pixels_to_inches
        total_length_inch = horizontal_length_inch + vertical_length_inch
        
        total_info = f"Bar {bar_number}: Total Length(pixels): {bar_info['total_length']:.2f}px, Total Length(inches): {total_length_inch:.2f}"
        horizontal_info = f"Horizontal Length: Pixels: {bar_info['horizontal_length']:.2f}, Inches: {horizontal_length_inch:.2f}"
        vertical_info = f"Vertical Length: Pixels: {bar_info['vertical_length']:.2f}, Inches: {vertical_length_inch:.2f}"
        
        cv2.putText(bar_image, total_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(bar_image, horizontal_info, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(bar_image, vertical_info, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
        return bar_image
    
    return None


def classify_bars(image_path, bar_info, center_line_height):
    
    classified_bars = []
    for bar in bar_info:
        h_y1 = bar["horizontal_bar"][1]
        if h_y1 < center_line_height:
            bar_type = "Top Steel"
        else:
            bar_type = "Bottom Steel"
        
        bar["type"] = bar_type
        classified_bars.append(bar)
    
    return classified_bars


def get_bars(image_path, masked_image, center_line_height, horizontal_pixel_length, horizontal_actual_length, vertical_pixel_length, vertical_actual_length, output_dir="public/bars"):

    os.makedirs(output_dir, exist_ok=True)

    # Clear the output directory if it exists
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    image = cv2.imread(masked_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(blur)[0]

    bar_info = []
    
    if lines is not None:
        horizontal_lines, vertical_lines, slanted_lines = segment_lines(lines)

        horizontal_lines = remove_duplicate_lines(horizontal_lines, is_horizontal=True)
        vertical_lines = sort_lines(vertical_lines)

        bar_info = detect_bars(horizontal_lines, vertical_lines, image.shape)

        # Calculate the conversion constants
        h_pixels_to_inches = horizontal_actual_length / horizontal_pixel_length
        v_pixels_to_inches = vertical_actual_length / vertical_pixel_length

        bar_count = 1
        classified_bars = classify_bars(image_path, bar_info, center_line_height)
        for bar in classified_bars:
            bar_image = draw_bar(image, bar, bar_count, h_pixels_to_inches, v_pixels_to_inches)
            if bar_image is not None:
                output_path = os.path.join(output_dir, f'bar_{bar_count}.png')
                cv2.imwrite(output_path, bar_image)
                # print(f"Image 'bar_{bar_count}.png' created successfully.")
                bar_count += 1
    else:
        print("No lines detected in the image.")

    return classified_bars


if __name__ == "__main__":
    image_path = "Sample Image Path"
    horizontal_pixel_length = 100  # Example value, replace with actual measurement
    horizontal_actual_length = 12  # Example value in inches, replace with actual measurement
    vertical_pixel_length = 100    # Example value, replace with actual measurement
    vertical_actual_length = 10    # Example value in inches, replace with actual measurement
    bar_info = get_bars(image_path, horizontal_pixel_length, horizontal_actual_length, vertical_pixel_length, vertical_actual_length)
    print(bar_info)
