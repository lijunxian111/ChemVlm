# -*- coding: GBK -*-
from concurrent.futures import ThreadPoolExecutor
import os
import random
import numpy as np
import json
from tqdm import tqdm
import re
from clean import md

def filter_func(question:str): 
    """
    function for filter out images not exists
    """
    image_exist = 1
    image_exist_in_q = 1
    match_pattern = r'!\[\]\([0-9a-f]+\)'
    images_in_q = re.findall(match_pattern, question['q_main'])
    if len(question['options'])!=0:
        options = ''.join(question['options'])
    else:
        options = ""
    for k,v in question['img_list'].items():
        
        if os.path.exists("/mnt/hwfile/ai4chem/share/data/"+question['img_list'][k]['raw_path']):
            pass
        else:
           image_exist = 0
           if k in question['q_main'] or k in options:
                image_exist_in_q = 0
           break

    return image_exist, image_exist_in_q

    
def convert(data_path:str):
    with open(data_path, 'r') as f:
        data_line = f.readlines()
    
    print(len(data_line))
    q_type_list = list()
    cnt_total = 0
    cnt_exist = 0
    cnt_exist_in_question = 0
    valid_images_all = []
    valid_images_in_q = []
    for i in tqdm(range(len(data_line))):
        line = data_line[i]
        question = json.loads(line)
        for k, v in question['img_list'].items():
            question['img_list'][k]['raw_path'] = question['img_list'][k]['raw_path'].replace('s3://llm-private-datasets/','').replace('s3://','').replace('p2/exam','p2-exam')
        #q_type_list.append(question['q_type'])
        #if i==73628:
        image_exist, image_exist_in_q = filter_func(question)
        if image_exist == 1:
            valid_images_all.append(question)
        elif image_exist == 0 and image_exist_in_q == 1:
            valid_images_in_q.append(question)
        cnt_total += 1
        cnt_exist += image_exist
        cnt_exist_in_question += image_exist_in_q
    print(f"Total num:{cnt_total}")
    print(f"{cnt_exist} questions can find all images, percentage: {cnt_exist/cnt_total*100}%")
    print(f"{cnt_exist_in_question} questions can find images in q, percentage: {cnt_exist_in_question/cnt_total*100}%")
    
    
    with open('/mnt/petrelfs/zhangdi1/lijunxian/datagen/valid.jsonl','w') as f:
        for item in valid_images_all:
            f.write(json.dumps(item)+"\n")

    with open('/mnt/petrelfs/zhangdi1/lijunxian/datagen/valid_in_q.jsonl','w') as f:
        for item in valid_images_in_q:
            f.write(json.dumps(item)+"\n")
    
    
    #print(list(set(q_type_list)))




"""
def read_and_convert_jsonl(data_paths: str, new_data_path: str):
    
    data_line = []
    
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            data_line += f.readlines()
    
    mm_json_list = []
    txt_json_list = []

    def worker(i):
        nonlocal mm_json_list,txt_json_list
        question = json.loads(data_line[i])
        if 'mllm-raw-media-p2/exam' in question:
            question = question.replace('mllm-raw-media-p2/exam','mllm-raw-media-p2-exam')
        #print(question)
        q_id = i
        if not '![]' in data_line[i]:
            conv = build_text_conversations(question)
            json_dict = {'id': str(q_id), 'conversations':conv}
            txt_json_list.append(json_dict)
        
        else:
            conv, images_list = build_mm_conversations(question)
            json_dict = {'id': 'mm'+str(q_id), 'images':images_list, 'conversations':conv}
            for i in images_list:
                if not os.path.exists('/mnt/hwfile/ai4chem/share/data/'+i):
                    return
            mm_json_list.append(json_dict)
        
        print(len(mm_json_list)+len(txt_json_list),len(data_line))


    with ThreadPoolExecutor() as executor:
        executor.map(worker,range(len(data_line)))
    
    with open('mm_pure_new.jsonl','w') as f:
        for item in mm_json_list:
            f.write(json.dumps(item)+"\n")

    with open('text_pure_new.jsonl','w') as f:
        for item in txt_json_list:
            f.write(json.dumps(item)+"\n")
"""


if __name__ == "__main__":
    #with open("/root/s3://chemllm/shiti-fengchao/edit/0910D3CFE6C887D221043E1D15D6904B.png",'r') as f:
        #pass
    #read_and_convert_jsonl(['/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/part-6629f71c08a4-000000.jsonl','/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/part-6629f7a32aad-000000.jsonl'],'mm_pure.jsonl')
    convert('/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/part-6629f71c08a4-000000.jsonl')