import pytesseract
from pdf2image import convert_from_path

# Convert PDF to images
pages = convert_from_path("document.pdf", 300)  # 300 DPI for better accuracy

# Extract text using Tesseract
text = "\n".join([pytesseract.image_to_string(page) for page in pages])

print(text)  # View extracted text


# import pytesseract
# from pdf2image import convert_from_path

# # Convert PDF to images
# images = convert_from_path("document.pdf", dpi=300)

# # Run OCR on each page
# for page_num, img in enumerate(images):
#     text_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
#     for i in range(len(text_data["text"])):
#         if text_data["text"][i].strip():
#             print(f"Text: {text_data['text'][i]}, BBox: ({text_data['left'][i]}, {text_data['top'][i]}, {text_data['width'][i]}, {text_data['height'][i]})")
