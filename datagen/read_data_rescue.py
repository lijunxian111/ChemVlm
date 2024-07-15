# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor
import os
import random
import numpy as np
import json
from tqdm import tqdm
import re
from clean import md

imgae_slots = ['start','mid','end']

def add_imgae_token_to_question(question,slot):
    if slot == 'start':
        question = '<image>\n'+question
    elif slot == 'mid':
        if '【A】' not in question and 'A.' not in question: #非选择题，中间位置不生效
            question = '<image>\n'+question
        else:
            question = question.replace('【A】','<image>\n【A】').replace('A.','<image>\nA.')
    elif slot == 'end':
        question = question + '\n<image>'

    return question


def random_option():
    styles = ['【','.']
    style = random.choice(styles)
    if style == '【':
        return {0:'【A】', 1:'【B】', 2:'【C】', 3:'【D】'}
    else:
        return {0:'A.', 1:'B.', 2:'C.', 3:'D.'}

def random_daan():
    styles = ['【','答案：','这一题的答案是']
    style = random.choice(styles)
    if style == '【':
        return '【答案】','【解析】','【考点】'
    elif style == '答案：':
        return '答案：\n','解析：\n','考点：\n'
    else:
        return '这一题的答案是','\n这题的解析是','\n这一题的主要考点是'


def build_text_conversations(question: dict):
    """
    build a human-gpt conversation in the prompt
    """
    conv = []
    
    question_body = question['q_main']
    human_prompt = question_body
    daan,jiexi,kaodian = random_daan()

    if len(question['options'])!=0:
        dict_ans = random_option()
        if 'A' not in question['options'][0] and (question['q_type']=='单选题' or question['q_type']=='多选题'):
            options = question['options']
            for i in range(len(options)):
                options[i] = dict_ans[i] + options[i]
                #options[0] = 'A.' + options[0]
                #options[1] = 'B.' + options[1]
                #options[2] = 'C.' + options[2]
                #options[3] = 'D.' + options[3]
        else:
            options = question['options']
        options = '\n'+'; '.join(options)+';\n'
        human_prompt += options


    conv_human = {'from':'human', 'value': md(human_prompt)}
    conv.append(conv_human)


    std_ans = question['std_ans'][0]
    ans_detail = question['answer_detail']
    if len(std_ans)!=0:
        gpt_prompt = daan + std_ans + '\n' + jiexi+ans_detail
    else:
        gpt_prompt = jiexi + ans_detail

    if 'keypoint' in question.keys():

        ky = '\n'+ kaodian + ', '.join(question['keypoint'])
        gpt_prompt += ky

    conv_gpt = {'from':'gpt','value':md(gpt_prompt)}
    conv.append(conv_gpt)

    return conv

def build_mm_conversations(question:dict):
    """
    build a human-gpt conversation in the prompt
    """

    match_pattern = r'!\[\]\([0-9a-f]+\)'
    replace_text = '\n<image>\n'
    daan,jiexi,kaodian = random_daan()

    conv = []
    question_body = question['q_main']
    human_prompt = question_body

    if len(question['options'])!=0:
        dict_ans = random_option()
        if 'A' not in question['options'][0] and (question['q_type']=='单选题' or question['q_type']=='多选题'):
            options = question['options']
            for i in range(len(options)):
                options[i] = dict_ans[i] + options[i]
                #options[0] = 'A.' + options[0]
                #options[1] = 'B.' + options[1]
                #options[2] = 'C.' + options[2]
                #options[3] = 'D.' + options[3]
        else:
            options = question['options']
        options = '\n'+'; '.join(options)+';\n'
        human_prompt += options

    if len(question['img_list']) == 1:
        human_prompt = re.sub(match_pattern, '', human_prompt)
    else:
        human_prompt = re.sub(match_pattern, '\n特殊占位符\n', human_prompt)
        if human_prompt.startswith('\n'):
            human_prompt = human_prompt[1:]
        if human_prompt.endswith('\n'):
            human_prompt = human_prompt[:-1]

    human_prompt = md(human_prompt)

    if len(question['img_list']) == 1:
        human_prompt = add_imgae_token_to_question(human_prompt,random.choice(imgae_slots))
    else:
        human_prompt = human_prompt.replace('特殊占位符', '<image>')

    conv_human = {'from':'human', 'value': human_prompt}

    conv.append(conv_human)


    std_ans = question['std_ans'][0]
    ans_detail = question['answer_detail']
    if len(std_ans)!=0:
        gpt_prompt = daan + std_ans + '\n' + jiexi+ans_detail
    else:
        gpt_prompt = jiexi + ans_detail

    if 'keypoint' in question.keys():

        ky = '\n'+ kaodian + ', '.join(question['keypoint'])
        gpt_prompt += ky

    gpt_prompt = re.sub(match_pattern, '...', gpt_prompt)

    conv_gpt = {'from':'gpt','value':md(gpt_prompt)}
    conv.append(conv_gpt)

    images_list = []
    for k,v in question['img_list'].items():
        images_list.append(question['img_list'][k]['raw_path'].replace('s3://llm-private-datasets/','').replace('s3://',''))

    return conv, images_list

    
def read_q_type(data_path:str):
    with open(data_path, 'r') as f:
        data_line = f.readlines()
    
    print(len(data_line))
    q_type_list = list()
    for i in range(len(data_line)):
        line = data_line[i]
        question = json.loads(line)
        
        #q_type_list.append(question['q_type'])
        if i==73628:
            print(question)
            break
    
    print(list(set(q_type_list)))


def read_and_convert_jsonl(data_paths: str, new_data_path: str):
    
    data_line = []
    
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            data_line += f.readlines()
    
    print(len(data_line))
    
    mm_json_list = []
    txt_json_list = []

    def worker(i):
        nonlocal mm_json_list,txt_json_list
        data_string = data_line[i]

        if 'mllm-raw-media-p2/exam' in data_string:
            data_string = data_string.replace('mllm-raw-media-p2/exam','mllm-raw-media-p2-exam')

        question = json.loads(data_string)

        for i in question['img_list']:
            if not os.path.exists('/mnt/hwfile/ai4chem/share/data/'+question['img_list'][i]['raw_path'].replace('s3://llm-private-datasets/','').replace('s3://','')):
                question['img_list'].pop(i)
                print('yes')
        
        if len(question['img_list']) == 0:
            print(data_string)

        q_id = i
        if len(question['img_list']) == 0:
            conv = build_text_conversations(question)
            json_dict = {'id': str(q_id), 'conversations':conv}
            txt_json_list.append(json_dict)
        
        else:
            conv, images_list = build_mm_conversations(question)
            json_dict = {'id': 'mm'+str(q_id), 'images':images_list, 'conversations':conv}

            mm_json_list.append(json_dict)
        
        print(len(mm_json_list)+len(txt_json_list),len(data_line))


    with ThreadPoolExecutor() as executor:
        executor.map(worker,range(len(data_line)))
    
    with open('mm_pure_rescue.jsonl','w') as f:
        for item in mm_json_list:
            f.write(json.dumps(item)+"\n")


if __name__ == "__main__":
    #with open("/root/s3://chemllm/shiti-fengchao/edit/0910D3CFE6C887D221043E1D15D6904B.png",'r') as f:
        #pass
    # read_and_convert_jsonl(['/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/part-6629f71c08a4-000000.jsonl','/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/part-6629f7a32aad-000000.jsonl'],'mm_pure.jsonl')
    # read_q_type('mm_pure.jsonl')
    read_and_convert_jsonl(['/mnt/petrelfs/zhangdi1/lijunxian/datagen/valid_in_q.jsonl.bak'],'mm_pure.jsonl')
