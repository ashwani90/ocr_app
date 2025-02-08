{
  "id": "doc1",
  "words": ["Invoice", "Number", ":", "12345"],
  "bboxes": [[30, 50, 100, 80], [110, 50, 170, 80], [180, 50, 190, 80], [200, 50, 280, 80]],
  "labels": ["O", "B-INVOICE_NUMBER", "O", "I-INVOICE_NUMBER"]
}

from transformers import LayoutLMv2ForTokenClassification, LayoutLMv2Processor
import torch

# Load LayoutLM model and processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=5)

# Label mapping (modify as per your dataset)
label_map = {"O": 0, "B-INVOICE_NUMBER": 1, "I-INVOICE_NUMBER": 2, "B-DATE": 3, "I-DATE": 4}

from torch.utils.data import Dataset

class InvoiceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = processor(item["words"], boxes=item["bboxes"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

        labels = [label_map[label] for label in item["labels"]]
        encoding["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoding

# Load dataset
train_data = InvoiceDataset(your_data)  # `your_data` is the formatted JSON dataset

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./layoutlm_model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    logging_dir="./logs",
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data
)

trainer.train()


def extract_fields(image):
    text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    words, boxes = [], []
    for i in range(len(text_data["text"])):
        if text_data["text"][i].strip():
            words.append(text_data["text"][i])
            boxes.append([text_data["left"][i], text_data["top"][i], text_data["width"][i], text_data["height"][i]])

    # Process through LayoutLM
    inputs = processor(words, boxes=boxes, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

    return [(words[i], predictions[0][i].item()) for i in range(len(words))]

# Test on a new document
test_image = convert_from_path("new_invoice.pdf")[0]
extracted_data = extract_fields(test_image)
print(extracted_data)


import json

json_output = {
    "invoice_number": "",
    "date": "",
    "total_amount": ""
}

for word, label in extracted_data:
    if label == label_map["B-INVOICE_NUMBER"] or label == label_map["I-INVOICE_NUMBER"]:
        json_output["invoice_number"] += word + " "
    elif label == label_map["B-DATE"] or label == label_map["I-DATE"]:
        json_output["date"] += word + " "

json_output = {k: v.strip() for k, v in json_output.items()}  # Clean up whitespace
print(json.dumps(json_output, indent=4))


