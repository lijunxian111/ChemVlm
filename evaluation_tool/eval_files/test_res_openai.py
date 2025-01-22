# -*- coding: utf8 -*-
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

from llama_3_performance import gen_all
from minicpm_performance import gen_results
from qwen2_vl_performance import gen_results_qwen_vl
from deepseek_performance import gen_all_results_deepseek




ds_collections = {
    'smiles_ocr':{
        'root': '',
        'question': '../../datagen/mm_chem_ocr.jsonl.test.jsonl',
        'prompt': "请根据指令回答一个SMILES化学分子式, 请只回答SMILES化学分子式而不输出其他内容",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'mm_gaokao_test_past': {
        'root': '/mnt/hwfile/ai4chem/share/data',
        'question': '../../datagen/mm_pure_fix.jsonl.test.jsonl',
        #'prompt': "请判断这道是什么题目并回答, 选择题请只回答A, B, C或D; 填空题请按顺序依次填空,并只回答填入的内容; 主观题请回答问题并给出详细步骤",
        'prompt': "请正确回答这道题目, 选择题请只回答A, B, C或D; 填空题请按顺序依次填空,并只回答填入的内容",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'CMMU':{
        'root': '',
        'question': '../../datagen/CMMU_test_no_multiple.jsonl',
        'prompt': "请正确回答这道题目, 选择题请只回答A, B, C或D; 填空题请按顺序依次填空,并只回答填入的内容",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'SciQA':{
        'root': '',
        'question': '../../SciQA/sciqa_test.jsonl',
        'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'SciQA_all':{
        'root': '',
        'question': '../../datagen/sciqa_test_all.jsonl',
        'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'Chembench_mol2cap': {
       'root': '',
       'question': '../../datagen/chembench_mol2caption.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_property': {
       'root': '',
       'question': '../../datagen/chembench_property.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_name_conv': {
       'root': '',
       'question': '../../datagen/chembench_name_conv.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_retro': {
       'root': '',
       'question': '../../datagen/chembench_retro.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_temperature': {
       'root': '',
       'question': '../../datagen/chembench_temperature.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_solvent': {
       'root': '',
       'question': '../../datagen/chembench_solvent.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_yield': {
       'root': '',
       'question': '../../datagen/chembench_yield.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chebi_caption':{
       'root': '',
       'question': '../../datagen/cheBI_caption_eng_test.jsonl',
       'prompt': "Please follow the question and give your answer. Answer briefly in English. Here is one example answer: \n'''It seems to me the molecule is an {} in which {}. It has a role as an {}. It is an {}, {}, {}. It derives from a {}.'''\n. Follow this format.",
       'max_new_tokens': 512,
       'min_new_tokens': 1,
    },
    'chirality_yon': {
       'root': '',
       'question': '/mnt/hwfile/ai4chem/hao/data_processing/chirality_mol_folder_new/charity_test_yon_modified.jsonl',
       'prompt': "You only need to answer whether the molecule in the picture is chiral.Your answer should only contain Yes or No.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'chirality_num': {
       'root': '',
       'question': '/mnt/hwfile/ai4chem/hao/data_processing/chirality_mol_folder_new/charity_easy_test.jsonl',
       'prompt': "You only need to answer the number of chiral centers.Your answer should only contain a number. For example 2 or 0.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'molecule_judge':{
        'root': '',
        'question': '/mnt/hwfile/ai4chem/hao/data_processing/molecule_caption_folder_extracted/test_all.jsonl',
        'prompt': "You just need to answer yes or no.",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'molecule_num':{
        'root': '',
        'question': '/mnt/hwfile/ai4chem/hao/data_processing/molecule_caption_folder_extracted/test_all.jsonl',
        'prompt': "You just need to return a number.",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'orderly_type':{
       'root': '',
       'question': '../../datagen/orderly_test_type.jsonl',
       'prompt': "Please follow the question and give your answer. Choose one from '''{}'''. Remember that only return several words of the reaction type and do not return explanations.",
       'max_new_tokens': 200,
       'min_new_tokens': 1,
    },
    'pistachio_type_sample':{
       'root': '',
       'question': '../../datagen/pistachio_mm_test_1000.jsonl',
       'prompt': "Please follow the question and give your answer. Choose one from '''{}'''. Remember that only return several words of the reaction type and do not return explanations.",
       'max_new_tokens': 200,
       'min_new_tokens': 1,
    },
    'orderly_product':{
       'root': '',
       'question': '../../datagen/orderly_test_project.jsonl',
       'prompt': "Please answer a SMILES string representing the molecule only, and do not answer any other things.",
       'max_new_tokens': 200,
       'min_new_tokens': 1,
    },
    

}

import requests

ss = requests.session()
ss.keep_alive = False

dashscope.api_key = 'sk-' #qwen API-KEY


openai.api_key = 'sk-'


with open('../../eval_results/gpt-4o-pistachio_type_sample.jsonl', 'r', encoding='utf8') as f:
    type_data = f.readlines()

f.close()
all_types = [json.loads(item)['annotation'] for item in type_data]
#all_types.remove('sub')
all_types = json.dumps(list(set(all_types)))
ds_collections['pistachio_type_sample']['prompt'] = ds_collections['pistachio_type_sample']['prompt'].format(all_types)


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
        client = OpenAI(api_key='sk-', base_url="https://api.claudeshop.top/v1")  #ChemVLM
        #client = OpenAI(api_key='sk-', base_url="https://api.claudeshop.top/v1")
        #client = OpenAI(api_key='sk-')
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

    

def get_answer(model_name, dataset_name, q_path, restore_file_path):
    data_list = list()
    with open(q_path, 'r') as f:
        data_to_test = f.readlines()
    
    if "molecule_judge" in dataset_name:
        data_to_test = data_to_test[0:250]
    if "molecule_num" in dataset_name:
        data_to_test = data_to_test[250:]
        
    for index, line in tqdm(enumerate(data_to_test)):
        line = line.replace('\\', '\\\\')
        try:
            line = json.loads(line)
        except:
            continue

        
        prompt = line['conversations'][0]['value'].replace('<image>\n','') + ds_collections[dataset_name]['prompt']
        
        #try:
        if 'images' in line.keys() or 'image' in line.keys():
            if 'images' in line.keys():
                img_path = [ds_collections[dataset_name]['root'] + line['images'][i] for i in range(len(line['images']))]
                if len(line['images']) == 1:
                    img_path = img_path[0]
            else:
                img_path = ds_collections[dataset_name]['root'] + line['image']
            #try:
            try:
                ans = call_multimodal(model_name, img_path, prompt)
                print(f"Answer: {ans}")
            
                data_list.append({'id': line['id'], 'text': ans, 'annotation': line['conversations'][1]['value']})
                if "qwen" in model_name or 'gpt' in model_name:
                    sleep(0.1)
            except:
                pass
        else:
            pass
        #except:
        #break
    
    writer = open(restore_file_path, 'w')
    for item in data_list:
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")

if __name__ == "__main__":
    #TODO: CMMU + Yi
    #dataset_names = ["Chembench_temperature","Chembench_retro","Chembench_solvent","Chembench_yield"]
    #dataset_names = ["chirality_yon","chirality_num","molecule"]
    dataset_names = ['pistachio_type_sample']
    model_names = ['gpt-4o']  #llama_3_2, gpt-4o/v, llava, yi-vl-plus/yi-vision, qwen-vl, glm-4v, gemini(testing)
    #test_special_prompt = None
    test_special_prompt = 'Please first conduct reasoning step by step, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.'
    if "qwen_2_vl" in model_names:
        if test_special_prompt is not None:
            extra_prompt = test_special_prompt

        else:
            extra_prompt = ds_collections[dataset_name]['prompt']
        
        for dataset_name in dataset_names:
            gen_results_qwen_vl(ds_collections[dataset_name]['question'], f'./eval_results/qwen_2_vl-{dataset_name}.jsonl', extra_prompt)

        model_names.remove('qwen_2_vl')
    
    if "deepseek" in model_names:
        for dataset_name in dataset_names:
            gen_all_results_deepseek(ds_collections[dataset_name]['question'], f'./eval_results/deepseek-vl-{dataset_name}.jsonl', ds_collections[dataset_name]['prompt'])

        model_names.remove('deepseek')

    if "llama_3_2" in model_names: 
        temperature = 0.7 #can be modified
        top_p = 0.9 #can be modified
        model_path = '/mnt/hwfile/ai4chem/share/llama_3_1'
        for dataset_name in dataset_names:
            gen_all(ds_collections[dataset_name]['question'], f'./eval_results/llama_3_2-{dataset_name}.jsonl', temperature, top_p, model_path, False, ds_collections[dataset_name]['prompt'])

        model_names.remove('llama_3_2')
    
    if "minicpm" in model_names: 
        for dataset_name in dataset_names:
            gen_results(ds_collections[dataset_name]['question'], f'./eval_results/minicpm-{dataset_name}.jsonl', ds_collections[dataset_name]['prompt'])

        model_names.remove('minicpm')

    if "llava" in model_names:
        
        command = "srun -p AI4Phys --gres=gpu:1 python ../../LLaVA/vqn_chem.py"
        args = shlex.split(command)
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = process.poll()
        model_names.remove('llava')
        #generate_answers("../../LLaVA/llava-v1.5-13b", ds_collections[dataset_name]['question'], ds_collections[dataset_name]['question']['root'], f'./{model_name}-{dataset_name}.jsonl')
    #else:
    if len(model_names)!=0:
        for model_name in model_names:
            for dataset_name in dataset_names:
                get_answer(model_name, dataset_name, ds_collections[dataset_name]['question'], f'./eval_results/{model_name}-{dataset_name}.jsonl')
