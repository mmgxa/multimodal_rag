from typing import Union, Optional, List, Sequence, Tuple
import base64
import requests
import os
from openai import OpenAI
import fitz

url = "http://localhost:8000/predict"


from chromadb.api.types import (
    DataLoader,
    Documents,
    Images,
    EmbeddingFunction,
    Embeddings,
)


def convert_pdf_to_images(pdf_file: str, image_dir: str) -> int:
    """
    Convert a PDF file into a series of images and save them in the specified directory.

    Parameters:
    - pdf_file (str): Path to the PDF file to be converted.
    - image_dir (str): Path to the directory where the images will be saved.

    Returns:
    - int: Number of saved images.

    """

    # create image_dir if it does not exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Open the PDF file
    pdf_document = fitz.open(pdf_file)
    num_pages = len(pdf_document)
    print(f"The document has {num_pages} pages")

    # Iterate through each page
    for page_number in range(num_pages):
        # Get the page
        page = pdf_document[page_number]

        # Convert the page to an image
        image = page.get_pixmap(alpha=False)

        # Save the image to a file
        image.save(f"{image_dir}/page_{page_number + 1}.jpg")

    # Close the PDF document
    pdf_document.close()

    return num_pages


class ImageBindEmbeddingFunction(EmbeddingFunction[Union[Documents, Images]]):
    def __init__(self) -> None:

        self.url = "http://localhost:8080"

    def _convert_to_base64(self, files: list) -> list:
        base64_encoded_files = []
        for file_path in files:
            base64_encoded_files.append(self._convert_file_to_base64(file_path))
        return base64_encoded_files

    def _convert_file_to_base64(self, file_path: str) -> str:
        with open(file_path, "rb") as binary_file:
            binary_file_data = binary_file.read()
            base64_encoded_data = base64.b64encode(binary_file_data)
            base64_message = base64_encoded_data.decode("utf-8")
            return base64_message

    def __call__(self, input: Union[Documents, Images]) -> Embeddings:
        embeddings: Embeddings = []
        # Here, we can check if the input is a list of files (i.e. images)
        # or a list of text
        if os.path.isfile(input[0]):
            req_body = {
                "images": self._convert_to_base64(input),
            }
            res = requests.post(self.url + "/vectorize", json=req_body)
            resBody = res.json()
            embeddings = resBody["imageVectors"]
        else:
            req_body = {
                "texts": input,
            }
            res = requests.post(self.url + "/vectorize", json=req_body)
            resBody = res.json()
            embeddings = resBody["textVectors"]

        return embeddings


class DummyLoader(DataLoader[List[Optional[str]]]):
    # def __init__(self, max_workers: int = 0) -> None:
    #     pass

    def __call__(self, uris: Sequence[Optional[str]]) -> List[Optional[str]]:
        return list(uris)


prompt_template: str = """Pretend that you are a helpful assistant that answers questions about content in a slide deck. 
Using only the information in the provided slide image answer the following question.
If you do not find the answer in the image then say I did not find the answer to this question in the slide deck.

{question}
"""


def _convert_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as binary_file:
        binary_file_data = binary_file.read()
        base64_encoded_data = base64.b64encode(binary_file_data)
        base64_message = base64_encoded_data.decode("utf-8")
        return base64_message


def query(question: str, multimodal_db) -> Tuple[str, str]:
    query_results = multimodal_db.query(
        query_texts=[question],
        n_results=5,
        include=["distances", "metadatas", "data", "uris"],
    )

    url = "http://localhost:8000/predict"

    # print(query_results)
    image = query_results["uris"][0][0]
    # print(f"Relevant image is: {image}")

    prompt = prompt_template.format(question=question)
    img_base64 = _convert_file_to_base64(image)
    data_uri = f"data:image/jpeg;base64,{img_base64}"

    data = {
        "image": data_uri,
        "question": prompt,
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "conv_mode": "llava_v1",
    }

    response = requests.post(
        url, headers={"Content-Type": "application/json"}, json=data
    )

    answer = response.json()

    return image, answer
