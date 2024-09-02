def segment_lines(lines, max_diff=5):
    """
    Categorizes a list of lines into horizontal, vertical, and slanted lines.

    Inputs:
    - lines: A list of lines, where each line is represented as [x1, y1, x2, y2].
    - max_diff: The maximum difference allowed between the x or y coordinates
      to consider a line as horizontal or vertical, respectively. Default is 5.

    Returns:
    - A tuple of three lists:
        - horizontal_lines: Lines that are classified as horizontal.
        - vertical_lines: Lines that are classified as vertical.
        - slanted_lines: Lines that are neither horizontal nor vertical.
    """
    
    # Initialize lists to store horizontal, vertical, and slanted lines
    horizontal_lines, vertical_lines, slanted_lines = [], [], []

    for line in lines:
        x1, y1, x2, y2 = line
        # Check if the line is vertical by comparing the x-coordinates
        if abs(x1 - x2) <= max_diff:
            vertical_lines.append([x1, y1, x2, y2])
        # Check if the line is horizontal by comparing the y-coordinates
        elif abs(y1 - y2) <= max_diff:
            horizontal_lines.append([x1, y1, x2, y2])
        else:
            # If the line is neither vertical nor horizontal, classify it as slanted
            slanted_lines.append([x1, y1, x2, y2])

    # Return the segmented lines categorized by type
    return horizontal_lines, vertical_lines, slanted_lines


def remove_duplicate_lines(lines, max_distance=20, max_y_diff=15, is_horizontal=True):
    """
    Removes duplicate or nearly overlapping lines by merging them into a single line.

    Inputs:
    - lines: A list of lines, where each line is represented as [x1, y1, x2, y2].
    - max_distance: The maximum allowable distance between the endpoints of two lines 
      for them to be considered as duplicates. Default is 20.
    - max_y_diff: The maximum allowable difference in y-coordinates for horizontal lines 
      (or x-coordinates for vertical lines) to be considered as duplicates. Default is 15.
    - is_horizontal: A boolean indicating whether to consider the lines as horizontal 
      (True) or vertical (False). Default is True.

    Returns:
    - A list of merged lines, where duplicates or nearly overlapping lines have been merged.
    """

    def merge_lines(lines, merged_lines):
        for line in lines:
            x1, y1, x2, y2 = line
            merged = False
            for i, merged_line in enumerate(merged_lines):
                x3, y3, x4, y4 = merged_line

                if is_horizontal:
                    # Merge lines if they are close enough horizontally and vertically
                    if ((abs(x1 - x4) < max_distance and abs(y1 - y4) < max_y_diff) or
                        (abs(x2 - x3) < max_distance and abs(y2 - y3) < max_y_diff)):
                        # Update the merged line to cover the full extent of both lines
                        merged_lines[i] = [min(x1, x3), y1, max(x2, x4), y2]
                        merged = True
                        break
                else:
                    # Merge lines if they are close enough vertically and horizontally
                    if ((abs(x1 - x3) < max_y_diff and abs(y1 - y3) < max_distance) or
                        (abs(x2 - x4) < max_y_diff and abs(y2 - y4) < max_distance)):
                        # Update the merged line to cover the full extent of both lines
                        merged_lines[i] = [x1, min(y1, y3), x2, max(y2, y4)]
                        merged = True
                        break
            
            # If the line was not merged with any existing line, add it as a new line
            if not merged:
                merged_lines.append([x1, y1, x2, y2])

        return merged_lines

    # Recursively merge lines until no more merges are possible
    merged_lines = merge_lines(lines, [])
    if len(lines) == len(merged_lines):
        return merged_lines
    else:
        return remove_duplicate_lines(merged_lines, max_distance, max_y_diff, is_horizontal)
