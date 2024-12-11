import requests
import openai

from openai import OpenAI
import os
import base64
import json

import dashscope
from dashscope import Generation
from dashscope import MultiModalConversation
from tqdm import tqdm
from PIL import Image

import google.generativeai as genai
from meutils.io.image import image_to_base64
#from LLaVA.vqn_chem import generate_answers

from zhipuai import ZhipuAI
import subprocess
import shlex

import re

from time import sleep

ss = requests.session()
ss.keep_alive = False

dashscope.api_key = 'sk-' #qwen API-KEY


openai.api_key = 'sk-'

def get_image_extension(image_path):
    return os.path.splitext(image_path)[1]
 
def call_multimodal(model_name, image_path, text):
    """
    call qwen-vl-v1, gpt-4 and other
    """
    local_file_path = image_path
    if "qwen" in model_name:
        messages = [{
            'role': 'system',
            'content': [{
                'text': 'You are a helpful assistant.'
            }]
        }, {
            'role':
                'user',
            'content': [
                {
                    'image': 'file://' + local_file_path
                },
                {
                    'text': text
                },
            ]
        }]
        # response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1, messages=messages)
        response = MultiModalConversation.call(model=model_name, messages=messages)

        return response.output.choices[0].message.content
    elif "gpt" in model_name:
        client = OpenAI(api_key='sk-', base_url="https://")  #ChemVLM
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        text = text.replace('<image>','')
        text = text.replace('\n','')
        if image_path is not None:
            if isinstance(image_path, list):
                for path in image_path:
                    base64_image = encode_image(path)
                    try:
                        img_type = get_image_extension(path)[1:]
                    except:
                        img_type = "png"
                    image_message = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{img_type};base64,{base64_image}"}
                    }
                    messages[0]["content"].append(image_message)
            else:
                # image = Image.open(image_path)
                base64_image = encode_image(image_path)
                try:
                    img_type = get_image_extension(image_path)[1:]
                except:
                    img_type = "png"
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{img_type};base64,{base64_image}"}
                }
                messages[0]["content"].append(image_message)

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=128
        )
        return response.choices[0].message.content
    
    elif "yi" in model_name:
        client = OpenAI(api_key='', base_url="https://api.lingyiwanwu.com/v1")
        #client = OpenAI(api_key='', base_url="https://api.lingyiwanwu.com/v1")
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        if image_path is not None:
            # image = Image.open(image_path)
            base64_image = encode_image(image_path)
            try:
                img_type = get_image_extension(image_path)[1:]
            except:
                img_type = "png"
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/{img_type};base64,{base64_image}"}
            }
            messages[0]["content"].append(image_message)

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=96
        )
        return response.choices[0].message.content
    
    elif 'gemini' in model_name:

        genai.configure(api_key='')
        image = Image.open(image_path)
        image_base64 = image_to_base64(image_path, for_image_url=False)
        try:
            img_type = get_image_extension(image_path)[1:]
        except:
            img_type = "png"
        contents_chat = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": text
                    },
                    {
                        "inline_data": {
                            "mime_type": f"image/{img_type}",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]

        client = genai.GenerativeModel(model_name)
        """
        response = client.generate_content(
            contents=contents_chat
        )
        """
        response = client.generate_content([image, text])
        print(response.text)
        return response.text
    
    elif "glm" in model_name:

        client = ZhipuAI(api_key="") # APIKey
        base64_image = encode_image(image_path)
        try:
            img_type = get_image_extension(image_path)[1:]
        except:
            img_type = "png"
        response = client.chat.completions.create(
            model=model_name,  # your model name
            messages=[
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url" : f"data:image/{img_type};base64,{base64_image}"
                    }
                }
                ]
            }
            ]
            )
        return response.choices[0].message.content




# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
