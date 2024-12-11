# -*- coding: utf8 -*-
import openai

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
import re
from typing import Optional
from test_res_openai import call_multimodal


dashscope.api_key = 'sk-eea7c876461747c5a6eebe0531164767' #qwen API-KEY
"""
client = OpenAI(
    api_key="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"
)
"""
 
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


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

def test_chemvl_perform_single_choice(data_to_eval: dict, q_info, extracted_gt: Optional = None):
    """
    evaluate single choice
    """
    #prompt_template = "你是一位熟悉题目答案和评分的专家，下面有一个对于选择题的回答:```{}```，请根据文本或```is```, ```Answer```,```answer```, ```Solution```, ```choice```, ```option```等字样的提示抽取这个回答认为的选项。注意，你只需要回答一个代表选项的字母, 如A,B,C,D" #一个or多个字母
    prompt_template = "你是一位熟悉化学题目答案和评分的专家，下面有一个对于选择题的回答:```{}```，请根据文本和```答案```字样的提示抽取这个回答给出的选项。注意，你只需要回答一个代表选项的字母, 如A,B,C,D" #一个or多个字母
    human_question = prompt_template.format(data_to_eval['text'])
    ground_truth = prompt_template.format(data_to_eval['annotation'])

    data_to_eval['annotation'] = data_to_eval['annotation'].replace('(','').replace(')','')
    
    #if len(data_to_eval['annotation']) == 1:
        #data_to_eval['text'] = data_to_eval['text'][0] only for special cases

    

    if len(data_to_eval['text'])<8:
        ans = "".join(re.findall('[A-Z]', data_to_eval['text']))

    
    elif "{'answer':" in data_to_eval['text']:
        ans = "".join(re.findall('[A-Z]', data_to_eval['text']))

    elif "解析" in data_to_eval['text']:
        remove_index = data_to_eval['text'].find("解析")
        ans = data_to_eval['text'][0: remove_index-1]
        ans = "".join(re.findall('[A-Z]', ans))

    elif "答案" in data_to_eval['text'] and "解析" not in data_to_eval['text']:
        search_index = data_to_eval['text'].find("答案")
        ans = data_to_eval['text'][search_index:]
        ans = "".join(re.findall('[A-Z]', ans))

    else:
        ans = call_multimodal('gpt-4o', None, human_question)
    
    
    if len(data_to_eval['annotation'])==1:
        std_ans = data_to_eval['annotation']
    elif extracted_gt is not None and len(extracted_gt['ground_truth']) == 1:
        std_ans = extracted_gt['ground_truth']
    elif "解析" in data_to_eval['annotation']:
        remove_index = data_to_eval['annotation'].find("解析")
        std_ans = data_to_eval['annotation'][0: remove_index-1]
        std_ans = "".join(re.findall('[A-Z]', ans))
    else:
        #std_ans = call_qwen(ground_truth)
        std_ans = call_multimodal('gpt-4o', None, ground_truth)

    #print(ans)
    if ans == std_ans:
        return 1, ans, std_ans
    else:
        return 0, ans, std_ans

def test_chemvl_perform_fill_in_blank(data_to_eval: dict, q_info, extracted_gt: Optional = None):
    """
    evaluate the fill-in-the-blank problem
    """
    prompt_template = "你是一位熟悉化学题目答案和评分的专家，下面有一个对于填空题的回答:```{}```, 请根据文本和```答案```字样的提示抽取这个回答给出的答案。注意,你只需要回答每个空的正确答案。"
    #judge_score_template = "你是一位熟悉化学题目答案和评分的专家, 下面有一个填空题```{}```和对于填空题的回答```{}```, 请根据标准答案```{}```来逐空给这道题目打分。注意, 这道题目满分为1分, 请按照正确的空的数目按比例给分, 注意各个空用' '或'；'等隔开。请只回答一个0-1(包括0,1)之间的数字而不输出任何其他内容。"
    judge_score_template = "你是一位熟悉题目评分的专家, 下面有一个题目的回答```{}```, 请根据标准答案```{}```来给这道题目打分。注意, 这道题目满分为1分, 如正确或基本正确请只回复数字1，否则请只回复数字0"
    human_question = prompt_template.format(q_info, data_to_eval['text'])
    if "解析" in data_to_eval['text']:
        remove_index = data_to_eval['text'].find("解析")
        ans = data_to_eval['text'][0: remove_index-1]
    else:   
        ans = data_to_eval['text']
    #ground_truth = prompt_template.format(q_info, data_to_eval['annotation'])
    #std_ans = call_qwen(ground_truth)
    std_ans = data_to_eval['annotation']
    if extracted_gt is not None:
        std_ans = extracted_gt['ground_truth']
    #print(std_ans)
    #judge_question = judge_score_template.format(q_info, ans, std_ans)
    judge_question = judge_score_template.format(ans, std_ans)
    
    #score = call_qwen(judge_question)
    score = call_multimodal('gpt-4o', None, judge_question)
    print(score)

    return float(score), ans, std_ans


def test_chemvl_perform(model_name, question_paths:list, ans_paths: list, gt_paths: Optional = None, task=""):
    """
    test our model's performance by qwen-max or gpt-4o
    """
    template = "请根据题目```{}```判断这道题目的题型, 请回答选择题, 填空题或主观题"
    ans_list = list()
    total_val_score = 0.
    total_right_score = 0.
    for idx, (q_path, answer_path) in tqdm(enumerate(zip(question_paths, ans_paths)), desc="evaluating gaokao"):
        if gt_paths is not None:
            with open(gt_paths[idx], 'r') as rd:
                data_gt = rd.readlines()
        else:
            data_gt = None
        with open(answer_path, 'r') as f:
            data_to_test = f.readlines()
        with open(q_path, 'r') as fp:
            origin_data = fp.readlines()
        total_q_num = 0.0
        cnt_right_num = 0
        cur_score = 0.0
        for index, (line, ori_line) in enumerate(zip(data_to_test, origin_data[:len(data_to_test)])):
            if data_gt is not None:
                line_gt = json.loads(data_gt[index])
            else:
                line_gt = None
            line = json.loads(line)
            ori_line = json.loads(ori_line)
            res = line['text']
            ori_q = ori_line['conversations'][0]['value']
            #ori_q = ori_line['question']
            #if "<image>" in ori_q:
            if True:
                """
                if "mm_pure" in question_paths[0]:
                    q = template.format(ori_q)  #only gaokao data need this step
                    ans_for_type = call_qwen(q)
                else:
                    ans_for_type = ""
                """
                #sleep(0.5)
                #if "选择" in ans_for_type:
                if ("question_id" in line.keys() and "xuan" in line['question_id']) or (len(re.findall('[A-Z]', line['annotation'])) == 1 and len(line['annotation'])==1 and line['annotation'].isupper()) or (line_gt is not None and len(line_gt['ground_truth'])==1):
                
                    score, llm_ans, gt_ans = test_chemvl_perform_single_choice(line, ori_q, line_gt)
                    cur_score += float(int(score))
                    total_q_num += 1
                    ans_comp = {'generated':llm_ans, 'ground_truth':gt_ans, 'score':score}
                    print(ans_comp)
                    ans_list.append(ans_comp)
                    sleep(0.5)
                    
                elif len(line['annotation'])>1:
                    try:
                        line_gt = None
                        score, llm_ans, gt_ans = test_chemvl_perform_fill_in_blank(line, ori_q, line_gt)
                        cur_score += float(int(score))
                        total_q_num += 1
                        ans_comp = {'generated':llm_ans, 'ground_truth':gt_ans, 'score': int(score)}
                        ans_list.append(ans_comp)
                        print(ans_comp)
                        sleep(0.5)
                    except:
                        pass
                else:
                    pass
            else:
                pass
                  
        if total_q_num != 0:
            total_val_score += total_q_num
            total_right_score += cur_score
            print(total_right_score/total_val_score)
        f.close()
        fp.close()
        try:
            rd.close()
        except:
            pass
    
    print(total_val_score)
    print(total_right_score)
    writer = open(f'/mnt/petrelfs/zhangdi1/lijunxian/eval_results/{model_name}_{task}.jsonl', 'w')
    for item in ans_list:
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')
    writer.close()
    
    
    print(f"总分{total_val_score}, 模型获得{total_right_score}分")

if __name__ == "__main__":
    #gaokao_chemvl_results = ['/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results/gaokao_chemvl_ft_6_4_0-merge__jia.jsonl',
                             #'/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results/gaokao_chemvl_ft_6_4_0-merge__jia1.jsonl',
                             #'/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results/gaokao_chemvl_ft_6_4_0-merge__xinkebiao.jsonl']
    #gaokao_chemvl_results = ['/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results/exam_200CKPT_chemvl_ft_6_19_0_merged_CMMU.jsonl']
    #llm_results = ['/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results/exam_213_pretrained_InternVL-Chat-V1-5_SciQA.jsonl']
    #origin_data = ['/mnt/petrelfs/zhangdi1/lijunxian/SciQA/sciqa_test.jsonl']
    model_name = "chemvlm26B-latest"
    task = "cmmu"
    
    
    llm_results = ['/mnt/petrelfs/zhangdi1/lijunxian/eval_results/result_chemvlm26B_1120__CMMU.jsonl']
    origin_data = ['/mnt/petrelfs/zhangdi1/lijunxian/datagen/CMMU_test_no_multiple.jsonl']

    """
    llm_results = ['/mnt/petrelfs/zhangdi1/lijunxian/eval_results/exam_pretrained_InternVL-Chat-V1-5_CMMU_pol.jsonl']
    origin_data = ['/mnt/petrelfs/zhangdi1/lijunxian/datagen/CMMU_test_politics.jsonl']
    """
    #gt_path = ['/mnt/petrelfs/zhangdi1/lijunxian/eval_results/chemvlm-new-26B_mmcr.jsonl']
    store_path = f'/mnt/petrelfs/zhangdi1/lijunxian/eval_results/{model_name}_{task}.jsonl'
    #llm_results = ['/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results/caption_chemvl_ft_8_01_0_checkpoint-22_Chembench_mol2cap.jsonl']
    #origin_data = ['']
    
    #test_chemvl_perform(model_name, origin_data, llm_results, gt_path, task)
    test_chemvl_perform(model_name, origin_data, llm_results, None, task)
    #test_chemvl_perform(model_name, origin_data, llm_results, None)
    #with open('/mnt/petrelfs/zhangdi1/lijunxian/datagen/mm_pure_fix.jsonl.test.jsonl','r') as f:
        #data = f.readlines()
    
    #print(json.loads(data[0]))
