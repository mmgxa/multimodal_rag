import requests
import json
import base64

# Assuming the FastAPI app is running on localhost:8000
url = "http://localhost:8000/predict"


def _convert_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as binary_file:
        binary_file_data = binary_file.read()
        base64_encoded_data = base64.b64encode(binary_file_data)
        base64_message = base64_encoded_data.decode("utf-8")
        return base64_message


image = "./car1.jpeg"
img_base64 = _convert_file_to_base64(image)
data_uri = f"data:image/jpeg;base64,{img_base64}"

print(img_base64[:10], img_base64[-10:])

# Prepare the request data
data = {
    "image": data_uri,
    "question": "What is the image about?",
    "max_new_tokens": 1024,
    "temperature": 0.1,
    "conv_mode": "llava_v1",
}

# Make the POST request
response = requests.post(url, headers={"Content-Type": "application/json"}, json=data)



# Check the response
if response.status_code == 200:
    print("Request successful!")
    print(response.json())
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)
