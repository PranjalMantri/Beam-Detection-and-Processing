from src.detection.beam_segmentor import detect_beams
from src.pdf_to_image.pdf_to_image import pdf_to_image
from src.detection.horizontal_scale_detector import detect_and_save_horizontal
from src.detection.vertical_scale_detector import  detect_and_save_vertical
from src.detection.color_comp import get_colors
from src.detection.detect_roi import create_image_mask
from src.data.bars import process_image

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
    if colors:
        print("These are the most prominent colors in the image: ", *list(set(colors)))

    beam_colour = input("Enter the colour of beam in your image: ").lower().title()
    column_colour = input("Enter the colour of column in your image: ").lower().title()

    if beam_colour not in colors:
        print("Invalid colour, please enter a colour that is in the image")
    elif column_colour not in colors:
        print("Invalid colour, Please enter a colour that is in the image")
    else:
        break

# print(f"Beam colour is {beam_colour}")
# print(f"Column colour is {column_colour}")

sample_beam_image = "public/Beams/beam_1.png"

coloured_beam = create_image_mask(sample_beam_image, beam_colour.lower(), output_dir="public/beam-image")
coloured_column = create_image_mask(sample_beam_image, column_colour.lower(), output_dir="public/column-image")

bar_info = process_image(coloured_beam)

# print(bar_info)