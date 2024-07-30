from src.detection.beam_segmentor import detect_beams
from src.pdf_to_image.pdf_to_image import pdf_to_image
from src.detection.horizontal_scale_detector import detect_and_save_horizontal
from src.detection.vertical_scale_detector import  detect_and_save_vertical

pdf_path = "public/pdfs/Class-1/PDF 3.pdf"

image_path = pdf_to_image(pdf_path)
# image_path = "public/images/Class 1/PDF 3_1.png"
beams = detect_beams(image_path)
horizontal_scale = detect_and_save_horizontal(image_path)
vertical_scale = detect_and_save_vertical(image_path)

if beams:
    print(f"Beams were detected successfully")

if horizontal_scale:
    print("Horizontal scales were detected successfully")

if vertical_scale:
    print("Vertical scales were detected successfully")