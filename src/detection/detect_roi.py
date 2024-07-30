import cv2
import numpy as np
import os

def create_image_mask(image_path, color, output_dir="public/roi"):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Delete all existing images in the directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for colors
    if color == 'red':
        lower_bound1 = np.array([0, 100, 100])
        upper_bound1 = np.array([10, 255, 255])
        lower_bound2 = np.array([160, 100, 100])
        upper_bound2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
        mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == 'cyan':
        lower_bound = np.array([80, 100, 100])
        upper_bound = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
    else:
        raise ValueError("Color not supported. Use 'red' or 'cyan'.")
    
    # Apply the mask to the original image
    color_only = cv2.bitwise_and(img, img, mask=mask)
    
    cropped_color_only = color_only[5:-5, 5:-5]
    
    # Save the new image
    output_image_path = os.path.join(output_dir, f"masked_image_{color}.png")
    write = cv2.imwrite(output_image_path, cropped_color_only)
    
    if write:
        print(f"Saved the image at {output_image_path}")
        return output_image_path
    
    return False

if __name__ == "__main__":
    image_path = "output_rectangles/rectangle_11.png"
    create_image_mask(image_path, 'red')  # For red color
    # create_image_mask(image_path, 'cyan')  # For cyan color
