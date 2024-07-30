import fitz  # PyMuPDF
import os

def pdf_to_image(pdf_path, output_dir="public/test-image", dpi=300):
    """
    Converts a one-page PDF to an image while maintaining high quality and preserving the PDF's name.

    Parameters:
    pdf_path (str): The path to the PDF file.
    output_dir (str): The directory where the output image will be saved.
    dpi (int): The resolution of the output image in DPI (dots per inch). Default is 300.
    """
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Ensure the PDF has at least one page
    if pdf_document.page_count < 1:
        print('The PDF does not contain any pages.')
        return

    # Get the first page
    page = pdf_document.load_page(0)

    # Set the zoom factor based on DPI
    zoom = dpi / 72  # 72 is the default DPI for PDFs

    # Define transformation matrix for zooming
    mat = fitz.Matrix(zoom, zoom)

    # Render page to a pixmap (image)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # Extract the PDF name and change the extension to .png
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_image_path = os.path.join(output_dir, f"{pdf_name}.png")

    # Save the image
    print(output_image_path)
    pix.save(output_image_path)
    print(f'Image saved to {output_image_path}')

    return output_image_path


if __name__ == "__main__":
    pdf_path = "public/pdfs/Class-1/PDF 3.pdf"
    # TODO: Add Proper output directory pathh
    output_dir = ""

    pdf_to_image(pdf_path, output_dir, dpi=400)
