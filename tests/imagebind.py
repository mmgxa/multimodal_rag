import base64
from typing import List
import requests


def _convert_to_base64(files: list) -> list:
    base64_encoded_files = []
    for file_path in files:
        base64_encoded_files.append(_convert_file_to_base64(file_path))
    return base64_encoded_files


def _convert_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as binary_file:
        binary_file_data = binary_file.read()
        base64_encoded_data = base64.b64encode(binary_file_data)
        base64_message = base64_encoded_data.decode("utf-8")
        return base64_message


url = "http://localhost:8080"
input = ["./car1.jpeg"]
req_body = {
    "images": _convert_to_base64(input),
}
res = requests.post(url + "/vectorize", json=req_body)
resBody = res.json()
print(len(resBody["imageVectors"][0]))  # should be 1024
