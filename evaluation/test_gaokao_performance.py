
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


dashscope.api_key = ''
 
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

def test_chemvl_perform_single_choice(data_to_eval: dict, q_info):
    """
    evaluate single choice
    """
    prompt_template = "你是一位熟悉化学题目答案和评分的专家，下面有一个对于选择题的回答:```{}```，请根据文本和```答案```字样的提示抽取这个回答给出的选项。注意，你只需要回答一个代表选项的字母" #一个or多个字母
    human_question = prompt_template.format(q_info, data_to_eval['text'])
    ground_truth = prompt_template.format(q_info, data_to_eval['annotation'])

    if len(data_to_eval['text'])<3:
        ans = "".join(re.findall('[A-Z]', data_to_eval['text']))
    
    elif "{'answer':" in data_to_eval['text']:
        ans = "".join(re.findall('[A-Z]', data_to_eval['text']))

    elif "解析" in data_to_eval['text']:
        remove_index = data_to_eval['text'].find("解析")
        ans = data_to_eval['text'][0: remove_index-1]
        ans = "".join(re.findall('[A-Z]', ans))
    else:
        ans = call_qwen(human_question)

    if len(data_to_eval['annotation'])==1:
        std_ans = data_to_eval['annotation']
    else:
        std_ans = call_qwen(ground_truth)

    #print(ans)
    if ans == std_ans:
        return 1, ans, std_ans
    else:
        return 0, ans, std_ans

def test_chemvl_perform_fill_in_blank(data_to_eval: dict, q_info):
    """
    evaluate the fill-in-the-blank problem
    """
    prompt_template = "你是一位熟悉化学题目答案和评分的专家，下面有一个对于填空题的回答:```{}```, 请根据文本和```答案```字样的提示抽取这个回答给出的答案。注意,你只需要回答每个空的正确答案。"
    judge_score_template = "你是一位熟悉化学题目答案和评分的专家, 下面有一个填空题```{}```和对于填空题的回答```{}```, 请根据标准答案```{}```来逐空给这道题目打分。注意, 这道题目满分为1分, 请按照正确的空的数目按比例给分, 注意各个空用' '或'；'隔开。请只回答一个0-1(包括0,1)之间的数字而不输出任何其他内容。"
    human_question = prompt_template.format(q_info, data_to_eval['text'])
    if "解析" in data_to_eval['text']:
        remove_index = data_to_eval['text'].find("解析")
        ans = data_to_eval['text'][0: remove_index-1]
    else:   
        ans = call_qwen(human_question)
    #ground_truth = prompt_template.format(q_info, data_to_eval['annotation'])
    #std_ans = call_qwen(ground_truth)
    std_ans = data_to_eval['annotation']
    #print(std_ans)
    judge_question = judge_score_template.format(q_info, ans, std_ans)
    
    score = call_qwen(judge_question)
    print(score)

    return float(score), ans, std_ans


def test_chemvl_perform(question_paths:list, ans_paths: list):
    """
    test our model's performance by qwen-max or gpt-4o
    """
    template = "请根据题目```{}```判断这道题目的题型, 请回答选择题, 填空题或主观题"
    ans_list = list()
    total_val_score = 0.
    total_right_score = 0.
    for q_path, answer_path in tqdm(zip(question_paths, ans_paths), desc="evaluating gaokao"):
        with open(answer_path, 'r') as f:
            data_to_test = f.readlines()
        with open(q_path, 'r') as fp:
            origin_data = fp.readlines()
        total_q_num = 0.0
        cnt_right_num = 0
        cur_score = 0.0
        for line, ori_line in zip(data_to_test, origin_data):
            line = json.loads(line)
            ori_line = json.loads(ori_line)
            res = line['text']
            ori_q = ori_line['conversations'][0]['value']
            #q = template.format(ori_q)
            #ans_for_type = call_qwen(q)
            #sleep(0.5)
            #if "选择" in ans_for_type:
            if len(line['annotation'])==1:
                try:
                    score, llm_ans, gt_ans = test_chemvl_perform_single_choice(line, ori_q)
                    cur_score += score
                    total_q_num += 1
                    ans_comp = {'generated':llm_ans, 'ground_truth':gt_ans, 'score':score}
                    print(ans_comp)
                    ans_list.append(ans_comp)
                    sleep(0.5)
                except:
                    pass
            elif len(line['annotation'])>1:
                try:
                    score, llm_ans, gt_ans = test_chemvl_perform_fill_in_blank(line, ori_q)
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
                  
        if total_q_num != 0:
            print(cnt_right_num/total_q_num)
            total_val_score += total_q_num
            total_right_score += cur_score
        f.close()
        fp.close()
    
    
    writer = open('', 'w')#写入地址
    for item in ans_list:
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')
    writer.close()
    
    
    print(f"总分{total_val_score}, 模型获得{total_right_score}分")

if __name__ == "__main__":
    
    llm_results = ['Chemvlm_SciQA.jsonl']
    origin_data = ['sciqa_test.jsonl']
    test_chemvl_perform(origin_data, llm_results)
