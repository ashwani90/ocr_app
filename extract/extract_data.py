import pytesseract
from pdf2image import convert_from_path

# Convert PDF to images
images = convert_from_path("images/invoice.pdf", dpi=300)

# Run OCR on each page
for page_num, img in enumerate(images):
    text_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    for i in range(len(text_data["text"])):
        if text_data["text"][i].strip():
            print(f"Text: {text_data['text'][i]}, BBox: ({text_data['left'][i]}, {text_data['top'][i]}, {text_data['width'][i]}, {text_data['height'][i]})")
