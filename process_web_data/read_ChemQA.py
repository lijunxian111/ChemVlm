# -*- coding: GBK -*-
import pandas as pd 
import numpy as np
import json
import os
import cv2
import yaml
from tqdm import tqdm


def read_data(path: str):
    """
    convert parquet to dict list
    """
    df = pd.read_parquet(path)
    #print(df['choices'].values[0])
    choices = [len(eval(df['choices'].values[i])) for i in range(len(df['choices'].values))]
    print(set(choices))
    print(np.unique(df['label'].values))
    data = df.to_dict('records') #转换为字典列表的关键句！！！

    return data

def convert_bytes_to_images(data, store_path: str, file_path: str):
    """
    make qa pairs
    """
    #print(data)
    question_list = list()
    dict_choices = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    for index, line in tqdm(enumerate(data)):
        q_id = f"chemqa_{index}"
        question = line['question']
        options = eval(line['choices'])  #非常重要
        ans = line['label']
        ans = dict_choices[ans]
        desc = line['description']
        if "train" in store_path:
            ans = "Answer: " + ans + ". " + desc
        else:
            pass
        #q_type = line['question_type']
        if index==0:
            print(ans)

        available_choices = "" 
        for i in range(len(options)-1):
            available_choices += (dict_choices[i] + "." + str(options[i]) + ' ')
        available_choices +=  (dict_choices[len(options)-1] + "." + str(options[-1]))

        if line['image'] is not None:
            question = '<image>\n' + question + available_choices   
            imgs = []
            #for i in range(1, 8):
                #if f"image {i}" in question or f"image {i}" in options:
            #print(line['image'])
            image_bytes = line['image']['bytes']
            # 将bytes数据转换为numpy矩阵
            image_np = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            #将numpy矩阵保存为jpg格式的图片文件
            cv2.imwrite(store_path + f"{index}.jpg", image_np)
            imgs.append(store_path + f"{index}.jpg")
        
            conversations = {'id': q_id, 'images': imgs, 'conversations':[{'from':'human', 'value': question}, {'from':'gpt', 'value': ans}]}
        else:
            question = question + available_choices
            conversations = {'id': q_id, 'conversations':[{'from':'human', 'value': question}, {'from':'gpt', 'value': ans}]}

        question_list.append(conversations)
        #image_bytes = img['bytes']
        #image_bytes = img
        # 将bytes数据转换为numpy矩阵
        #image_np = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 将numpy矩阵保存为jpg格式的图片文件
        #cv2.imwrite(store_path + f"{index}.jpg", image_np)

    writer = open(file_path, 'w')
    for item in question_list:
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")


    

if __name__ == "__main__":
    #print('A'+1)
    """
    raw_data_1 = read_data('ChemQA/data/train-00000-of-00002.parquet')
    raw_data_2 = read_data('ChemQA/data/train-00001-of-00002.parquet')
    raw_data = raw_data_1 + raw_data_2
    """
    raw_data = read_data('ChemQA/data/valid-00000-of-00001.parquet')
    convert_bytes_to_images(raw_data, chemqa/val/', 'datagen/chemqa_val.jsonl')
