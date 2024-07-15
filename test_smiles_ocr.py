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


def sim_cal_metric(input_smiles:str, gt_smiles:str):
    """
    The metric to calculate similarity of generated smiles and ground truth(tanimoto similarity)
    """
    input_mol = Chem.MolFromSmiles(input_smiles)
    gt_mol = Chem.MolFromSmiles(gt_smiles)
    if input_mol is None and gt_mol is not None:
        return 0.0
    elif gt_mol is None:
        return None
    reference_fp = Chem.RDKFingerprint(gt_mol)
    input_fp = Chem.RDKFingerprint(input_mol)
    tanimoto = DataStructs.FingerprintSimilarity(reference_fp, input_fp)
    return tanimoto

def call_qwen(question:str):
    """
    Call qwen-max to extract SMILES from generated texts and ground truth texts
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

def call_qwen_multimodal(image_path, text):
    """
    call qwen-vl-v1 
    """
    local_file_path = image_path
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
                'image': 'file://'+local_file_path
            },
            {
                'text': text
            },
        ]
    }]
    #response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1, messages=messages)
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
    
    return response.output.choices[0].message.content

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def test_internvl():
    """
    test raw internvl performance
    """
    tanimoto_list = list()

    with open('/mnt/petrelfs/zhangdi1/lijunxian/qwen_ocr.jsonl','r') as f:
        gt = f.readlines()
    with open('/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results/smiles_ocr_pretrained_InternVL-Chat-V1-5_smiles_ocr.jsonl','r') as fp:
        pred = fp.readlines()
    
    for i in tqdm(range(len(gt))):
        gt_line = json.loads(gt[i])
        pred_line = json.loads(pred[i])

        tanimoto_similarity = sim_cal_metric(pred_line['text'], gt_line['ground_truth'])
        if tanimoto_similarity is not None:
            tanimoto_list.append(tanimoto_similarity)
    
    average_tanimoto = sum(tanimoto_list)/len(tanimoto_list)

    origin_lenth = len(tanimoto_list)
    tanimoto_one_count = len(
        [tanimoto for tanimoto in tanimoto_list if tanimoto == 1.0]
    )
    tanimoto_to_one = tanimoto_one_count/origin_lenth
    
    print(f"平均相似度{average_tanimoto}, tanimoto@1.0为{tanimoto_to_one*100}%")

def test_qwen_vl(restore_qwen_ans_path:str):
    """
    test raw Qwen_vl performance
    """
    tanimoto_list = list()

    with open('/mnt/petrelfs/zhangdi1/lijunxian/qwen_ocr.jsonl','r') as f:
        gt = f.readlines()
    
    with open('/mnt/petrelfs/zhangdi1/lijunxian/datagen/mm_chem_ocr.jsonl.test.jsonl','r') as fp:
        data_to_test = fp.readlines()
    
    qwen_vl_smiles_list = list()

    for i in tqdm(range(len(data_to_test)), desc='Calling Qwen-VL to return SMILES'):
        single_data = json.loads(data_to_test[i])
        img_path = single_data['image']
        #img_path = '/mnt/petrelfs/zhangdi1/lijunxian/dog.jpeg'
        #prompt = "图片上描述了什么"
        prompt = "你是一位精通化学分子图识别和化学表达式的专家"+single_data['conversations'][0]['value'] + "注意，请只回答图上分子的化学表达式而不回答其他内容。"
        prompt = prompt.replace('\n','').replace('<image>','')
        ans_from_qwen = call_qwen_multimodal(img_path, prompt)
        print(ans_from_qwen)
        qwen_vl_smiles_list.append(ans_from_qwen)
        #break
    return
    
    with open(restore_qwen_ans_path, 'w', encoding='utf-8') as file:
        for data in gen_and_gt_list:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    f.close()
    fp.close()
    file.close()
    
    with open(restore_qwen_ans_path,'r') as f:
        pred = f.readlines()
    
    for i in tqdm(range(len(gt)), desc='Calculating Qwen-VL performance'):
        gt_line = json.loads(gt[i])
        pred_line = json.loads(pred[i])

        tanimoto_similarity = sim_cal_metric(pred_line['text'], gt_line['ground_truth'])
        if tanimoto_similarity is not None:
            tanimoto_list.append(tanimoto_similarity)
    
    average_tanimoto = sum(tanimoto_list)/len(tanimoto_list)

    origin_lenth = len(tanimoto_list)
    tanimoto_one_count = len(
        [tanimoto for tanimoto in tanimoto_list if tanimoto == 1.0]
    )
    tanimoto_to_one = tanimoto_one_count/origin_lenth
    
    print(f"平均相似度{average_tanimoto}, tanimoto@1.0为{tanimoto_to_one*100}%")

def test_chemvl_perform_smiles_ocr(ans_paths: list, restore_path: str):
    """
    test our model's performance on SMILES OCR by qwen and rdkit fingerprint similarity
    """
    
    tanimoto_list = list()
    gen_and_gt_list = list()

    for answer_path in ans_paths:
        with open(answer_path, 'r') as f:
            data_to_test = f.readlines()

        for line in tqdm(data_to_test):
            line = json.loads(line)
            generated_res = line['text']
            ground_truth = line['annotation']
            human_prompt_gen = f'你是一位熟悉化学SMILES表示的专家。请提取下列语句中的SMILES：\n```'+generated_res+'```\n注意，你只需要返回一个SMILES式子。'
            human_prompt_gt = f'你是一位熟悉化学SMILES表示的专家。请提取下列语句中的SMILES：\n```'+ground_truth+'```\n注意，你只需要返回一个SMILES式子。'

            ans_gen = call_qwen(human_prompt_gen)
            ans_gt = call_qwen(human_prompt_gt)
            print(ans_gen)
            print(ans_gt)

            #sleep(1)
            gen_and_gt_list.append({'generated': ans_gen, 'ground_truth': ans_gt})

            tanimoto_similarity = sim_cal_metric(ans_gen, ans_gt)
            if tanimoto_similarity is not None:
                tanimoto_list.append(tanimoto_similarity)
    
    average_tanimoto = sum(tanimoto_list)/len(tanimoto_list)

    origin_lenth = len(tanimoto_list)
    tanimoto_one_count = len(
        [tanimoto for tanimoto in tanimoto_list if tanimoto == 1.0]
    )
    tanimoto_to_one = tanimoto_one_count/origin_lenth
    
    print(f"平均相似度{average_tanimoto}, tanimoto@1.0为{tanimoto_to_one*100}%")
    with open(restore_path, 'w', encoding='utf-8') as file:
        for data in gen_and_gt_list:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    eval_model_type = 'qwenvl' #internvl, chemvlm or qwenvl
    if eval_model_type == 'chemvlm':
        ocr_chemvl_results = ['/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results/smiles_ocr_2b_wxz_chemvl_2B_ft_7_3_0_merge_smiles_ocr.jsonl']
        restore_res_path = '/mnt/petrelfs/zhangdi1/lijunxian/qwen_ocr_2b.jsonl'
        test_chemvl_perform_smiles_ocr(ocr_chemvl_results, restore_res_path)
    elif eval_model_type == 'internvl':
        test_internvl()
    elif eval_model_type == 'qwenvl':
        qwen_path = '/mnt/petrelfs/zhangdi1/lijunxian/qwen_vl_ocr.jsonl'
        test_qwen_vl(qwen_path)
    else:
        raise ValueError(f"Cannot recognize {eval_model_type}")