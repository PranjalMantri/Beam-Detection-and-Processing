def calculate_intersection_over_union(rect1, rect2):
    x1, y1, width1, height1 = rect1
    x2, y2, width2, height2 = rect2

    x_intersection, y_intersection = max(x1, x2), max(y1, y2)
    x_intersection_end, y_intersection_end = min(x1 + width1, x2 + width2), min(y1 + height1, y2 + height2)

    intersection_width = x_intersection_end - x_intersection
    intersection_height = y_intersection_end - y_intersection

    if intersection_width <= 0 or intersection_height <= 0:
        return 0, 0

    intersection_area = intersection_width * intersection_height
    area_rect1 = width1 * height1
    area_rect2 = width2 * height2
    union_area = area_rect1 + area_rect2 - intersection_area

    iou = intersection_area / union_area
    return iou, intersection_width


def merge_rectangles(rect1, rect2):
    x1, y1, width1, height1 = rect1
    x2, y2, width2, height2 = rect2

    x_merged = min(x1, x2)
    y_merged = min(y1, y2)
    width_merged = max(x1 + width1, x2 + width2) - x_merged
    height_merged = max(y1 + height1, y2 + height2) - y_merged

    return (x_merged, y_merged, width_merged, height_merged)