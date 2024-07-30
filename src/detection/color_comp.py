from PIL import Image
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree

# List of General colours
COLOR_LIST = [
    ((0, 0, 0), "Black"),
    ((255, 255, 255), "White"),
    ((255, 0, 0), "Red"),
    ((0, 255, 0), "Green"),
    ((0, 0, 255), "Blue"),
    ((255, 255, 0), "Yellow"),
    ((0, 255, 255), "Cyan"),
    ((255, 0, 255), "Magenta"),
    ((128, 128, 128), "Gray"),
    ((128, 0, 0), "Maroon"),
    ((128, 128, 0), "Olive"),
    ((128, 0, 128), "Purple"),
    ((0, 128, 128), "Teal"),
]


def get_color_name(rgb_tuple):
    tree = KDTree([color[0] for color in COLOR_LIST])
    
    # Find the closest color
    _, index = tree.query(rgb_tuple)
    
    return COLOR_LIST[index][1]


def get_colors(image_path, num_colors=10, similarity_threshold=10):
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    pixels = list(img.getdata())

    color_counts = defaultdict(int)
    for pixel in pixels:
        color_counts[pixel] += 1
    
    # Merging colors that are similar
    merged_colors = defaultdict(int)
    for color, count in color_counts.items():
        found_similar = False
        for merged_color in merged_colors:
            if all(abs(c1 - c2) <= similarity_threshold for c1, c2 in zip(color, merged_color)):
                merged_colors[merged_color] += count
                found_similar = True
                break
        if not found_similar:
            merged_colors[color] = count
    
    # Sort by number of pixels
    sorted_colors = sorted(merged_colors.items(), key=lambda x: x[1], reverse=True)
    
    top_colors = sorted_colors[:num_colors]
    named_colors = [get_color_name(rgb) for rgb, _ in top_colors]
    
    return named_colors, top_colors

if __name__ == "__main__":
    image_path = "public/Beams/beam_0.png"
    named_colors, top_colors = get_colors(image_path)

    print("Detected colors:")
    for name, ((r, g, b), count) in zip(named_colors, top_colors):
        print(f"{name}: RGB({r}, {g}, {b}), Count: {count}")




#     ((192, 192, 192), "Silver"),
#     ((0, 128, 0), "Green"),
#     ((0, 0, 128), "Navy"),
#     ((244, 164, 96), "SandyBrown"),
#     ((210, 105, 30), "Chocolate"),
#     ((188, 143, 143), "RosyBrown"),
#     ((255, 69, 0), "OrangeRed"),
#     ((255, 140, 0), "DarkOrange"),
#     ((255, 215, 0), "Gold"),
#     ((184, 134, 11), "DarkGoldenrod"),
#     ((218, 165, 32), "Goldenrod"),
#     ((189, 183, 107), "DarkKhaki"),
#     ((240, 230, 140), "Khaki"),
#     ((154, 205, 50), "YellowGreen"),
#     ((85, 107, 47), "DarkOliveGreen"),
#     ((107, 142, 35), "OliveDrab"),
#     ((124, 252, 0), "LawnGreen"),
#     ((127, 255, 0), "Chartreuse"),
#     ((173, 255, 47), "GreenYellow"),
#     ((0, 255, 127), "SpringGreen"),
#     ((0, 250, 154), "MediumSpringGreen"),
#     ((143, 188, 143), "DarkSeaGreen"),
#     ((46, 139, 87), "SeaGreen"),
#     ((102, 205, 170), "MediumAquamarine"),
#     ((32, 178, 170), "LightSeaGreen"),
#     ((64, 224, 208), "Turquoise"),
#     ((72, 209, 204), "MediumTurquoise"),
#     ((175, 238, 238), "PaleTurquoise"),
#     ((0, 206, 209), "DarkTurquoise"),
#     ((95, 158, 160), "CadetBlue"),
#     ((70, 130, 180), "SteelBlue"),
#     ((176, 196, 222), "LightSteelBlue"),
#     ((176, 224, 230), "PowderBlue"),
#     ((173, 216, 230), "LightBlue"),
#     ((135, 206, 235), "SkyBlue"),
#     ((135, 206, 250), "LightSkyBlue"),
#     ((25, 25, 112), "MidnightBlue"),
#     ((65, 105, 225), "RoyalBlue"),
#     ((138, 43, 226), "BlueViolet"),
#     ((75, 0, 130), "Indigo"),
#     ((106, 90, 205), "SlateBlue"),
#     ((147, 112, 219), "MediumPurple"),
#     ((238, 130, 238), "Violet"),
#     ((218, 112, 214), "Orchid"),
#     ((255, 0, 255), "Fuchsia"),
#     ((255, 20, 147), "DeepPink"),
#     ((255, 105, 180), "HotPink"),
#     ((255, 182, 193), "LightPink"),
#     ((255, 192, 203), "Pink"),
#     ((250, 128, 114), "Salmon"),
#     ((255, 160, 122), "LightSalmon"),
#     ((255, 127, 80), "Coral"),
#     ((240, 128, 128), "LightCoral"),
#     ((255, 99, 71), "Tomato"),
#     ((205, 92, 92), "IndianRed"),
#     ((233, 150, 122), "DarkSalmon"),
#     ((250, 235, 215), "AntiqueWhite"),
#     ((255, 235, 205), "BlanchedAlmond"),
#     ((255, 228, 196), "Bisque"),
#     ((255, 222, 173), "NavajoWhite"),
#     ((245, 222, 179), "Wheat"),
#     ((222, 184, 135), "BurlyWood"),
#     ((210, 180, 140), "Tan"),
#     ((188, 143, 143), "RosyBrown"),
#     ((244, 164, 96), "SandyBrown")