

import spacy
from spacy.training.example import Example

TRAIN_DATA = [
    ("Invoice Number: 12345", {"entities": [(16, 21, "INVOICE_NUMBER")]}),
    ("Date: 2024-02-07", {"entities": [(6, 16, "DATE")]}),
    ("Total Amount: $500", {"entities": [(14, 18, "AMOUNT")]}),
]

nlp = spacy.blank("en")  # Create a blank language model
ner = nlp.add_pipe("ner")

# Add custom entity labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Train model
optimizer = nlp.begin_training()
for i in range(30):  # 30 iterations
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], sgd=optimizer)

# Save trained model
nlp.to_disk("custom_ner")

nlp = spacy.load("custom_ner")
doc = nlp("Invoice Number: 67890, Date: 2025-01-15, Total: $1200")

for ent in doc.ents:
    print(ent.label_, ":", ent.text)
