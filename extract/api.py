from fastapi import FastAPI, File, UploadFile
import torch
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io
import json

# Initialize FastAPI
app = FastAPI()

'''
This depends on LayoutLLM
so will need to check how LayoutLM works and how we can use them
'''

# Load LayoutLM model & processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForTokenClassification.from_pretrained("./layoutlm_model")  # Load fine-tuned model

# Label mapping (Ensure it matches your trained model)
label_map = {0: "O", 1: "B-INVOICE_NUMBER", 2: "I-INVOICE_NUMBER", 3: "B-DATE", 4: "I-DATE"}

# Function to extract text & bounding boxes
def extract_text_and_boxes(image):
    text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    words, boxes = [], []
    for i in range(len(text_data["text"])):
        if text_data["text"][i].strip():
            words.append(text_data["text"][i])
            boxes.append([text_data["left"][i], text_data["top"][i], text_data["width"][i], text_data["height"][i]])

    return words, boxes

# Function to process the document through LayoutLM
def extract_fields(image):
    words, boxes = extract_text_and_boxes(image)

    # Process through LayoutLM
    inputs = processor(words, boxes=boxes, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

    extracted_data = []
    for i in range(len(words)):
        label = label_map.get(predictions[0][i].item(), "O")
        extracted_data.append({"word": words[i], "label": label})

    return extracted_data

# API Endpoint for document processing
@app.post("/extract/")
async def extract_document(file: UploadFile = File(...)):
    # Convert PDF to image
    if file.filename.endswith(".pdf"):
        images = convert_from_path(io.BytesIO(await file.read()), dpi=300)
        image = images[0]  # Process first page only
    else:
        image = Image.open(io.BytesIO(await file.read()))

    # Extract structured data
    extracted_data = extract_fields(image)

    # Convert to JSON format
    json_output = {"invoice_number": "", "date": ""}
    for item in extracted_data:
        if "INVOICE_NUMBER" in item["label"]:
            json_output["invoice_number"] += item["word"] + " "
        elif "DATE" in item["label"]:
            json_output["date"] += item["word"] + " "

    json_output = {k: v.strip() for k, v in json_output.items()}  # Clean up whitespace
    return json_output