<div align="center">

# Multimodal RAG - ImageBind + ChromaDB + LLaVA

</div>

# Objective:
In this repository, RAG on a PDF presentation has been done using Imagebind (a multimodal embedding model), ChromaDB,  and LLaVA (a multimodal model).


# How does it work?
First, we convert a PDF file to a set of images in a directory. These images are passed to ImageBind, a multimodal embedding model that converts each image to a 1024-dim embedding vector. Note that using ImageBind, we can ingest all images at once.

We also pass along metadata that captures the location of the image being embedded.
## Serving Models
We can build the images to serve the model using FastAPI.
```
docker build -t rag:imagebind ./imagebind
docker build -t rag:llava ./llava
```

To serve the models, run

```
docker run -it --rm -p 8080:8080 rag:imagebind
docker run -it --rm --gpus all -p 8000:8000 rag:llava
```

To test whether the models are up-and-running, navigate to the tests directory and run
```
python imagebind.py # prints 1024, the size of embedding vector
python llava.py # prints the description of a car
```

# Ingest and Retrieve
We define helper functions in the `utils.py` file. The functions are:
- `convert_pdf_to_images`: This converts PDF to images using the `PyMuPDF` library
- `ImageBindEmbeddingFunction`: This is used by Chroma when ingesting the image. It simply makes a call to the ImageBind model.
- `DummyLoader`: ChromaDB forces the use of a dataloader when passing image paths. Since we don't want any pre-processing, we return the input data without modification.
- `query`: The query function used to invoke LLaVA along with a suitable prompt.


The complete notebook can be found [here](./notebook/rag_demo.ipynb).

