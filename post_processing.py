import re
import json

text = "Invoice No: 12345\nDate: 2024-02-07\nTotal: $500"
data = {
    "invoice_number": re.search(r"Invoice No:\s*(\d+)", text).group(1),
    "date": re.search(r"Date:\s*([\d-]+)", text).group(1),
    "total_amount": re.search(r"Total:\s*\$(\d+)", text).group(1),
}

print(data)

json_data = json.dumps(data, indent=4)
print(json_data)