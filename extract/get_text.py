import pytesseract
from pdf2image import convert_from_path

# Convert PDF to images
pages = convert_from_path("images/invoice.pdf", 300)  # 300 DPI for better accuracy

# Extract text using Tesseract
text = "\n".join([pytesseract.image_to_string(page) for page in pages])

print(text)  # View extracted text
