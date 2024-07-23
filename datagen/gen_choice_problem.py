# -*- coding: GBK -*-

from http import HTTPStatus
import dashscope
from dashscope import Generation
from dashscope import MultiModalConversation
from dashscope.api_entities.dashscope_response import Role

from rdkit import Chem
from rdkit import DataStructs
import os
import base64
import json
import time
import random
from tqdm import tqdm

from time import sleep

dashscope.api_key = 'sk-eea7c876461747c5a6eebe0531164767' #qwen API-KEY

def call_qwen(question:str):
    """
    Call qwen-max to answer the question
    """
    messages = [
        {'role': 'user', 'content': question}]
    response = Generation.call(
        model='qwen-max',
        max_tokens=128,
        messages=messages,
        result_format='message'
        #stream=True,
        #incremental_output=True
    )
    full_content = ''
    #for response in responses:
    if response.status_code == HTTPStatus.OK:
        full_content = response.output.choices[0]['message']['content']
        
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
    return full_content

def gen_wrong_choices(smiles: str):
    """
    use qwen-max to generate 3 wrong answers
    """
    templates = "你是一个化学SMILES专家。假设SMILES表达式{}是一道选择题的正确选项内容, 请根据化学键，手性，原子排列，原子种类和原子个数等特性帮我创造三个错误选项并输出。注意，你的回答必须包含三个SMILES字符串，且只需输出三个SMILES字符串而不输出任何其它内容。"
    
    question = templates.format(smiles)
    ans = call_qwen(question)
    #print(ans)
    return ans.split("\n")


def construct_multiple_choice_question(original_data_path, extracted_data_path, store_path):
    """
    use wrong answers from LLM to generate new question.
    original_data_path: the original question-answer pair
    extracted_data_path: smiles extracted by qwen
    """
    
    with open(original_data_path,'r') as f:
        eval_data = f.readlines()
    with open(extracted_data_path,'r') as fp:
        extracted_data = fp.readlines()
    f.close()
    fp.close()
    
    new_problems_list = list()
    existed_smiles_lst = list()
    templates = "A.{} B.{} C.{} D.{}。"
    idx = 0
    right_choices = ['A','B','C','D']

    for data_line, smiles_line in tqdm(zip(eval_data, extracted_data), desc="Construct choice problems"):
        right_pos = idx % 4
        idx += 1
        data_dict = json.loads(data_line)
        smiles_dict = json.loads(smiles_line)

        gt_smiles = smiles_dict['ground_truth']
        wrong_lst = gen_wrong_choices(gt_smiles)
        if len(wrong_lst)==2:
            fake_ans = gt_smiles[0:len(gt_smiles)//2]
            wrong_lst.append(fake_ans)
        elif len(wrong_lst) == 1:
            fake_ans = gt_smiles[0:len(gt_smiles)//2]
            random_ans = random.choice(existed_smiles_lst)
            wrong_lst.append(fake_ans)
            wrong_lst.append(random_ans)

        wrong_lst.insert(right_pos, gt_smiles)
        choices = templates.format(wrong_lst[0], wrong_lst[1], wrong_lst[2], wrong_lst[3])
        question = data_dict['conversations'][0]['value']
        if question[-1] == '<':
            question = question.replace('\n<image>','')
            question += choices
            question += '\n<image>'
        
        else:
            question += choices
        
        right_choice = right_choices[right_pos]

        data_dict['conversations'][0]['value'] = question
        data_dict['conversations'][1]['value'] = right_choice
        
        print(idx)
        new_problems_list.append(data_dict)
        existed_smiles_lst.append(gt_smiles)

    random.shuffle(new_problems_list)
    writer = open(store_path, 'w')
    for item in new_problems_list:
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')
    writer.close()
    print('Results saved to {}'.format(store_path))

if __name__ == "__main__":
    original_data_path = "/mnt/petrelfs/zhangdi1/lijunxian/datagen/mm_chem_ocr.jsonl.test.jsonl"
    extracted_data_path = "/mnt/petrelfs/zhangdi1/lijunxian/qwen_ocr.jsonl"
    store_path = "/mnt/petrelfs/zhangdi1/lijunxian/datagen/mm_chem_ocr_choice_problems.jsonl"

    construct_multiple_choice_question(original_data_path, extracted_data_path, store_path)

    