import requests

file_path = "invoice.pdf"
url = "http://localhost:8000/extract/"

with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())