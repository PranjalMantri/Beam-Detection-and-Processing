def segment_lines(lines, max_diff=5):
    horizontal_lines, vertical_lines, slanted_lines = [], [], []
    for line in lines:
        x1, y1, x2, y2 = line
        if abs(x1 - x2) <= max_diff:
            vertical_lines.append([x1, y1, x2, y2])
        elif abs(y1 - y2) <= max_diff:
            horizontal_lines.append([x1, y1, x2, y2])
        else:
            slanted_lines.append([x1, y1, x2, y2])
    return horizontal_lines, vertical_lines, slanted_lines


def remove_duplicate_lines(lines, max_distance=20, max_y_diff=15, is_horizontal=True):
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