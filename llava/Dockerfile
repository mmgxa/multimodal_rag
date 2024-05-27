FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Get requirements
COPY requirements.txt requirements.txt

# install git
RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install git -y \
  && rm -rf /var/lib/apt/lists/*

# Install any other needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Enable fast model download
ENV HF_HUB_ENABLE_HF_TRANSFER 1

RUN huggingface-cli download "anymodality/llava-v1.5-7b"

COPY . .

CMD ["python", "app.py"]
