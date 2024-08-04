from src.pdf_to_image.pdf_to_image import pdf_to_image
from src.detection.beam_segmentor import detect_beams
from src.detection.horizontal_scale_detector import detect_and_save_horizontal
from src.detection.vertical_scale_detector import  detect_and_save_vertical
from src.detection.color_comp import get_colors
from src.detection.detect_roi import create_image_mask
from src.data.bars import get_bars
from src.scale.get_vertical_scale import get_vertical_scale
from src.scale.get_horizontal_scale import get_horizontal_scale
from src.scale.parse_measurement_text import parse_measurement
from src.data.clean import clean_mask_image
from src.data.column import get_column_data
from src.data.beam_center import get_center_height

pdf_path = "public/pdfs/Class-1/PDF 3.pdf"

image_path = pdf_to_image(pdf_path)

beams = detect_beams(image_path)
horizontal_scale = detect_and_save_horizontal(image_path, add_padding=True)
vertical_scale = detect_and_save_vertical(image_path,add_padding=True)

if not beams:
    print("Something went wrong while detecting beams")

if not horizontal_scale:
    print("Something went wrong while detecing horizontal scale")

if not vertical_scale:
    print("Something went wrong while detecting vertical scale")

colors, _ = get_colors(image_path)

while True:
    # if colors:
    #     print("These are the most prominent colors in the image: ", *list(set(colors)))

    colors = ["red", "cyan"]

    # beam_colour = input("Enter the colour of beam in your image: ").lower().title()
    # column_colour = input("Enter the colour of column in your image: ").lower().title()

    beam_colour = "red"
    column_colour = "cyan"

    if beam_colour not in colors:
        print("Invalid colour, please enter a colour that is in the image")
    elif column_colour not in colors:
        print("Invalid colour, Please enter a colour that is in the image")
    else:
        break


sample_beam_image = "public/Beams/beam_5.png"

center_height = get_center_height(sample_beam_image)

vertical_line_length, vertical_scale_text = get_vertical_scale()
horizontal_line_length, horizontal_scale_text = get_horizontal_scale()

vertical_scale_in_inches = parse_measurement(vertical_scale_text[0][0])
horizontal_scale_in_inches = parse_measurement(horizontal_scale_text[0][0])

coloured_beam = clean_mask_image(create_image_mask(sample_beam_image, beam_colour.lower(), output_dir="public/beam-image"))
coloured_column = clean_mask_image(create_image_mask(sample_beam_image, column_colour.lower(), output_dir="public/column-image"), type="column")

bar_info = get_bars(sample_beam_image, coloured_beam, center_height, horizontal_line_length, horizontal_scale_in_inches, vertical_line_length, vertical_scale_in_inches)
column_info = get_column_data(coloured_column, center_height, horizontal_line_length, horizontal_scale_in_inches)

# print(f"Bar Information: {bar_info}")
print(f"Column Information: {column_info}")