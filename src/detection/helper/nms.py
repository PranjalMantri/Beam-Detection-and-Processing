def calculate_intersection_over_union(rect1, rect2):
    """
    Calculate the Intersection over Union (IoU) between two rectangles.

    Args:
        rect1 (tuple): A tuple representing the first rectangle (x, y, width, height).
        rect2 (tuple): A tuple representing the second rectangle (x, y, width, height).

    Returns:
        tuple: IoU score and the width of the intersection area.
    """
    x1, y1, width1, height1 = rect1
    x2, y2, width2, height2 = rect2

    # Calculate intersection coordinates
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x_intersection_end = min(x1 + width1, x2 + width2)
    y_intersection_end = min(y1 + height1, y2 + height2)

    # Calculate intersection width and height
    intersection_width = x_intersection_end - x_intersection
    intersection_height = y_intersection_end - y_intersection

    # If there's no intersection, return 0 IoU and width
    if intersection_width <= 0 or intersection_height <= 0:
        return 0, 0

    # Calculate areas
    intersection_area = intersection_width * intersection_height
    area_rect1 = width1 * height1
    area_rect2 = width2 * height2
    union_area = area_rect1 + area_rect2 - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou, intersection_width


def merge_rectangles(rect1, rect2):
    """
    Merge two rectangles into one that encompasses both.

    Args:
        rect1 (tuple): A tuple representing the first rectangle (x, y, width, height).
        rect2 (tuple): A tuple representing the second rectangle (x, y, width, height).

    Returns:
        tuple: A tuple representing the merged rectangle (x, y, width, height).
    """
    x1, y1, width1, height1 = rect1
    x2, y2, width2, height2 = rect2

    # Determine the coordinates and dimensions of the merged rectangle
    x_merged = min(x1, x2)
    y_merged = min(y1, y2)
    width_merged = max(x1 + width1, x2 + width2) - x_merged
    height_merged = max(y1 + height1, y2 + height2) - y_merged

    return (x_merged, y_merged, width_merged, height_merged)
