import requests
import base64
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware

from llava.model import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

#############################
import os
from logging import getLogger
from fastapi import FastAPI, Response, status
import uvicorn
from contextlib import asynccontextmanager

from pydantic import BaseModel

logger = getLogger("uvicorn")

model_id = "anymodality/llava-v1.5-7b"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, vision_tower, image_processor
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor
    logger.info("Model initialization complete")
    yield


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    image: str
    question: str
    max_new_tokens: int
    temperature: float
    conv_mode: str


class PredictionResponse(BaseModel):
    prediction: str


@app.post("/predict")
async def predict(data: PredictionRequest):
    image_file = data.image
    raw_prompt = data.question
    max_new_tokens = data.max_new_tokens
    temperature = data.temperature
    conv_mode = data.conv_mode if data.conv_mode else "llava_v1"

    # image_file = data.pop("image", data)
    # raw_prompt = data.pop("question", data)
    # max_new_tokens = data.pop("max_new_tokens", 1024)
    # temperature = data.pop("temperature", 0.1)
    # conv_mode = data.pop("conv_mode", "llava_v1")

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    inp = f"{roles[0]}: {raw_prompt}"
    inp = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    )
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif image_file.startswith("data:image"):
        _, encoded_data = image_file.split(",", 1)
        img_data = base64.b64decode(encoded_data)
        image = Image.open(BytesIO(img_data)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    disable_torch_init()
    image_tensor = (
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )

    keywords = [stop_str]
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    outputs = tokenizer.decode(
        output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    print(type(outputs))
    return outputs


if __name__ == "__main__":

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port="8000",
        log_level="info",
    )
