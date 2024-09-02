import cv2

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    Draw a bounding box on an image with optional label.

    Args:
        x (tuple): Coordinates of the bounding box (x1, y1, x2, y2).
        img (numpy.ndarray): Image on which to draw the bounding box.
        color (list, optional): Color of the bounding box in BGR format (default is red).
        label (str, optional): Label to draw above the bounding box.
        line_thickness (int, optional): Thickness of the bounding box lines (default is calculated based on image size).
    """
    # Set default line thickness if not provided
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1

    # Set default color if not provided
    color = color or [255, 0, 0]  # Default is red in BGR format

    # Define coordinates for the top-left (c1) and bottom-right (c2) corners of the bounding box
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    # Draw the rectangle on the image
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        # Set font thickness based on line thickness
        tf = max(tl - 1, 1)

        # Compute the size of the label text
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        # Calculate position for the background rectangle of the label
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

        # Draw the background rectangle for the label
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)

        # Put the label text on the image
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
