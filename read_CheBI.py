# -*- coding: GBK -*-
import pandas as pd 
import numpy as np
import json
import os
import cv2
import yaml
from tqdm import tqdm
import random


smiles_templates = [
    "What is the name of the molecule shown in this image?",
"What kind of compound does this molecular structure represent?",
"Can you help me identify this molecule in the image?",
"What is the chemical structure in this image?",
"Can you tell me the chemical formula of this molecule?",
"What kind of molecule is shown in the image?",
"Can you explain the composition of the molecule in the image?",
"What kind of chemical molecule is in this image?",
"Can you identify the molecule in the image?",
]

smiles_chinese_templates = [
    "请问这张图片展示的分子名称是什么？",
    "这个分子结构代表哪种化合物？",
    "能否帮我识别图中的这个分子？",
    "这张图像中的化学结构是什么？",
    "这个分子的化学式能告诉我吗？",
    "图中所示的是哪种分子？",
    "您能解释一下图片中分子的组成吗？",
    "这张图片中的是哪一种化学分子？",
    "您能识别出图片中的这个分子吗？"]



caption_templates = [
    "Can you provide a description of the molecule in the image?",
"How would you describe the molecule shown in the image?",
"Could you detail the molecule depicted in the image?",
"What are the characteristics of the molecule in the image?",
"Can you explain the features of the molecule in the image?",
"Could you elaborate on the molecule presented in the image?",
"How would you characterize the molecule shown in the image?",
"Can you outline the properties of the molecule in the image?",
"What details can you provide about the molecule in the image?",
"Can you give an overview of the molecule depicted in the image?"
]
 
iupac_templates = [
    "Can you provide the IUPAC name of the molecule depicted in the image?",
"What is the IUPAC nomenclature for the molecule in the image?",
"Could you tell me the IUPAC name for the molecule shown in the image?",
"What is the systematic IUPAC name of the molecule displayed in the image?",
"How is the molecule in the image named according to IUPAC standards?",
"What is the official IUPAC name of the molecule illustrated in the image?",
"Could you identify the IUPAC name of the molecule in the image?",
"What would the IUPAC name be for the molecule presented in the image?",
"Can you specify the IUPAC designation of the molecule shown in the image?",
"What is the formal IUPAC name for the molecule in the image?",
]

iupac_chinese_templates = [
"图中分子的IUPAC名称是什么？",
"你能告诉我图中分子的IUPAC命名吗？",
"图中所示化合物的IUPAC名称是什么？",
"这个图中分子的IUPAC标准名称是什么？",
"请问图中显示的分子的IUPAC命名是什么？",
"图中所示分子的系统IUPAC命名是什么？",
"图中分子的IUPAC正式名称是什么？",
"你能提供图中分子的IUPAC名称吗？",
"图中所示分子的IUPAC名称是什么？",
"图中显示的化学分子的IUPAC命名是怎样的？"
]

smiles_answer_templates = english_expressions = [
    "I believe the molecular formula in this image, when represented with SMILES, should be {}.",
    "From my perspective, the molecular structure in the image is written in SMILES format as {}.",
    "In my view, the molecular formula displayed in the image with a SMILES representation is {}.",
    "According to my understanding, the molecular formula in the image can be represented in SMILES as {}.",
    "I think the chemical structure in this image, when described with SMILES, is {}.",
    "From my personal perspective, the molecular structure in the image can be represented with SMILES as {}.",
    "As I observe, the molecular formula in this image, expressed in SMILES format, should be {}.",
    "My interpretation is that the molecular formula in this image, according to SMILES format, is {}.",
    "In my opinion, the SMILES expression of the molecular formula shown in this image should be {}.",
    "For me, the SMILES representation of the molecule in the image should be {}.",
    "My analysis shows that the molecular formula in this image is represented in SMILES format as {}.",
    "In my view, the molecular structure in this image is expressed in SMILES notation as {}.",
    "Based on my analysis, the molecular formula displayed in the image, if written in SMILES, would be {}.",
    "I guess the molecular formula in this image, if expressed in SMILES syntax, would be {}.",
    "From my angle, the molecular formula in the image in SMILES format is {}.",
    "I speculate that the molecular formula in this image, expressed in SMILES format, could be {}.",
    "Based on my observation, the molecular formula in this image, as per SMILES, is {}.",
    "I estimate that the molecular structure in this image, written in SMILES, should be {}.",
    "My opinion is that the molecular structure of the image in SMILES format is {}.",
    "In my view, the molecular formula in the image translated into SMILES language should be {}.",
    "From my perspective, the molecular structure in this image, represented in SMILES format, is {}.",
    "My understanding is that the molecular structure in the image, represented with SMILES code, is {}.",
    "From my vantage point, the molecular formula in the image, if represented with SMILES, would be {}.",
    "From my analysis, the molecular formula in this image expressed in SMILES syntax is {}.",
    "My judgment is that the molecular structure displayed in the image, written in SMILES, should be {}.",
    "I believe the chemical structure in this image, when converted to SMILES format, is {}.",
    "I feel that the molecular formula in this image, represented in SMILES fashion, is {}.",
    "I believe that the molecular structure in this chart, if represented with SMILES syntax, is {}.",
    "From a scientific viewpoint, the molecular formula in this image, marked with SMILES, should be {}.",
    "I am confident that the molecular structure shown in this image, expressed in SMILES format, would be {}."
]

chinese_smiles_answers = [
    "我认为这张图片里的分子式，用SMILES表示应为 {}。",
    "在我看来，图片中的分子结构用SMILES格式写作 {}。",
    "我的看法是，该图显示的分子式的SMILES表示法是 {}。",
    "根据我的理解，图中的分子式可用SMILES形式表示为 {}。",
    "我觉得，这个图片中的化学结构，用SMILES来描述，就是 {}。",
    "从我个人的角度出发，图片里的分子结构用SMILES可以表示为 {}。",
    "据我观察，此图中的分子式以SMILES形式表达，应为 {}。",
    "我的解读是，这幅图中的分子式，按SMILES格式是 {}。",
    "以我的见解，这张图片展示的分子式的SMILES表达应该是 {}。",
    "对我来说，该图片中分子的SMILES表示应是 {}。",
    "我的分析显示，这张图中的分子式以SMILES格式表示为 {}。",
    "在我看来，这张图片中的分子结构的SMILES表达式是 {}。",
    "依我分析，图中展示的分子式如果用SMILES写出来应是 {}。",
    "我猜想这幅图片的分子式，如果用SMILES语法来表达，会是 {}。",
    "从我的角度判断，图片中所示分子式的SMILES形式是 {}。",
    "我推测这张图片中的分子式，以SMILES格式表达，可能是 {}。",
    "依据我的观察，这个图片展示的分子式，按SMILES来说是 {}。",
    "我估计这幅图的分子结构，用SMILES方式写出来，应当是 {}。",
    "我的意见是，该图的分子结构以SMILES格式表示则为 {}。",
    "在我看来，图中的分子式转换成SMILES语言应该是 {}。",
    "依我之见，这个图片中的分子结构，SMILES格式表示为 {}。",
    "我的理解是，图中的分子结构，用SMILES代码表示，是 {}。",
    "从我的视角，该图片中的分子式，如果用SMILES表示，会是 {}。",
    "从我的分析来看，这张图片中的分子式用SMILES语法表达，为 {}。",
    "我的判断是，这幅图展示的分子结构，用SMILES来写应是 {}。",
    "我认为这张图片中的化学结构，转换成SMILES格式，就是 {}。",
    "我感觉这张图中的分子式，以SMILES的方式表示出来，是 {}。",
    "我认为这幅图表中的分子结构，用SMILES语法表示，就是 {}。",
    "从科学的视角来看，这张图片中的分子式用SMILES标记，应是 {}。",
    "我相信这张图片展示的分子结构，采用SMILES格式来表达，将是 {}。"
]

caption_answer_templates = [
    "As I see it, {}",
"To my mind, {}",
"In my view, {}",
"Personally, I think {}",
"In my assessment, {}",
"It seems to me {}",
"I believe that {}",
"To me, {}",
"As far as I'm concerned, {}"
]

iupac_answers = [
     "I think the molecular formula in this image, when given its IUPAC name, should be {}",
"In my opinion, the IUPAC name for the molecular formula in this image should be {}",
"I am convinced that the IUPAC name for the molecule shown in this image should be {}",
"It is my belief that the molecular formula depicted here, when named using IUPAC standards, should be {}",
"From my perspective, the molecular formula in this image should be named according to IUPAC as {}",
"I hold the view that the molecule in the image, when expressed with its IUPAC name, should be {}",
"It seems to me that the molecular formula shown in this image should have the IUPAC name {}",
"I consider that the IUPAC name for the molecular formula illustrated in this image should be {}",
"I feel that the IUPAC name for the molecule depicted in this image should be {}",
"To my mind, the molecular formula in this image, when converted to its IUPAC name, should be {}",
"In my assessment, the IUPAC name for the molecule in this image should be {}",
"I would suggest that the molecular formula shown here, when translated into IUPAC nomenclature, should be {}",
"My view is that the IUPAC name for the molecular formula in this image should be {}",
"I reckon that the molecule displayed in this image, when named using IUPAC conventions, should be {}",
"I assert that the IUPAC name for the molecular formula shown in this image should be {}",
"My perspective is that the molecular formula depicted in the image should be given the IUPAC name {}",
"I surmise that the IUPAC name for the molecule in the image should be {}",
"It is my opinion that the molecular formula in this image, when represented by its IUPAC name, should be {}",
"I maintain that the molecule in the image should be named with the IUPAC name {}",
"I gather that the IUPAC name for the molecular formula depicted in this image should be {}"
]

iupac_chinese_answers = [
    "我觉得图中分子的IUPAC命名是 {}",
"在我看来，图中分子的IUPAC命名是 {}",
"依我之见，图中分子的IUPAC命名是 {}",
"从我的角度来看，图中分子的IUPAC命名是 {}",
"对我而言，图中分子的IUPAC命名是 {}",
"我的看法是，图中分子的IUPAC命名是 {}",
"我的观点是，图中分子的IUPAC命名是 {}",
"我相信，图中分子的IUPAC命名是 {}",
"就我所知，图中分子的IUPAC命名是 {}",
"依我之见，图中分子的IUPAC名称是 {}",
"就个人而言，我认为图中分子的IUPAC命名是 {}",
"我个人的意见是，图中分子的IUPAC命名是 {}",
"我的理解是，图中分子的IUPAC命名是 {}",
"我推测图中分子的IUPAC命名是 {}",
"我的结论是，图中分子的IUPAC命名是 {}",
"在我看来，图中分子的IUPAC名称是 {}",
"我持有的观点是，图中分子的IUPAC命名是 {}",
"从我的立场来看，图中分子的IUPAC命名是 {}",
"我估计图中分子的IUPAC命名是 {}",
"我认为图中化合物的IUPAC命名是 {}"
]

def add_image_token(text):
    a = random.choice('01')
    if a == '0':
        text = '<image>\n' + text
    else:
        text = text + '\n<image>'
    return text

def get_prompt_from_templates(template):

    return add_image_token(random.choice(template))

def get_ans_from_templates(template, raw_ans):
    
    return random.choice(template).format(raw_ans)

def read_data(path: str):
    """
    convert parquet to dict list
    """
    df = pd.read_csv(path)
    print(df.columns)
    #print(len(df.values))
    #print(df['description'].values[0])
    #print(df['choices'].values[0])
    data = df.to_dict('records') #转换为字典列表的关键句！！！

    """
    for root, _, files in os.walk('/mnt/petrelfs/zhangdi1/lijunxian/CheBI/image'):
        files_names = files
        break
    """
    print(data[0].keys())
    return data

def gen_smiles_dataset(data, img_store_path: str):
    """
    make qa pairs of smiles
    """
    #print(data)
    prompt_list = list()
    for index, line in tqdm(enumerate(data), desc='gen smiles'):
        #print(line['CID'])
        q_id = 'chebi_smiles' + str(line['CID'])
        lang = random.choice(['english','chinese'])
        if lang == 'english':
            prompt = get_prompt_from_templates(smiles_templates)   
            ans_prompt = get_ans_from_templates(smiles_answer_templates, line['SMILES'])
        else: 
            prompt = get_prompt_from_templates(smiles_chinese_templates)   
            ans_prompt = get_ans_from_templates(chinese_smiles_answers, line['SMILES'])
        #q_type = line['question_type']
        if index==0:
            print(ans_prompt)
        
        imgs = []
        
        cid_num = line['CID']
        image_np = cv2.imread('/mnt/petrelfs/zhangdi1/lijunxian/CheBI/image/'+f'CID_{cid_num}.png')
        #将numpy矩阵保存为jpg格式的图片文件
        cv2.imwrite(img_store_path + f"{index}.png", image_np)
        imgs.append(img_store_path + f"{index}.png")
        
        conversations = {'id': q_id, 'images': imgs, 'conversations':[{'from':'human', 'value': prompt}, {'from':'gpt', 'value': ans_prompt}]}
        

        prompt_list.append(conversations)
    
    return prompt_list

def gen_caption_dataset(data, img_store_path: str):
    """
    make qa pairs of molecule captions
    """
    #print(data)
    prompt_list = list()
    for index, line in tqdm(enumerate(data), desc='gen caption'):
        q_id = 'chebi_caption' + str(line['CID'])
        prompt = get_prompt_from_templates(caption_templates)
        
        ans_prompt = get_ans_from_templates(caption_answer_templates, line['description'].replace('The', 'the'))
        #q_type = line['question_type']
        if index==0:
            print(ans_prompt)
            image_np = cv2.imread(img_store_path + f"{index}.png")
        
        imgs = []
     
        #image_np = cv2.imread(img_store_path + f"{index}.png")
        #将numpy矩阵保存为jpg格式的图片文件
        #cv2.imwrite(img_store_path + f"{index}.png", image_np) 只需要写入一次即可
        imgs.append(img_store_path + f"{index}.png")
        
        conversations = {'id': q_id, 'images': imgs, 'conversations':[{'from':'human', 'value': prompt}, {'from':'gpt', 'value': ans_prompt}]}
        

        prompt_list.append(conversations)
    
    return prompt_list

def gen_iupac_dataset(data, img_store_path: str):
    """
    make qa pairs of molecule captions
    """
    #print(data)
    prompt_list = list()
    for index, line in tqdm(enumerate(data), desc='gen iupac'):
        q_id = 'chebi_iupac' + str(line['CID'])
        lang = random.choice(['english','chinese'])
        if lang == 'english':
            prompt = get_prompt_from_templates(iupac_templates)
            ans_prompt = get_ans_from_templates(iupac_answers, line['iupacname'])
        else:
            prompt = get_prompt_from_templates(iupac_chinese_templates)
            ans_prompt = get_ans_from_templates(iupac_chinese_answers, line['iupacname'])
        #q_type = line['question_type']
        if index==0:
            print(ans_prompt)
            image_np = cv2.imread(img_store_path + f"{index}.png")
        
        imgs = []
     
        #image_np = cv2.imread(img_store_path + f"{index}.png")
        #将numpy矩阵保存为jpg格式的图片文件
        #cv2.imwrite(img_store_path + f"{index}.png", image_np) 只需要写入一次即可
        imgs.append(img_store_path + f"{index}.png")
        
        conversations = {'id': q_id, 'images': imgs, 'conversations':[{'from':'human', 'value': prompt}, {'from':'gpt', 'value': ans_prompt}]}
        

        prompt_list.append(conversations)
    
    return prompt_list


def write_total_data(prompt_list: list, file_path: str):
    writer = open(file_path, 'w')
    for item in prompt_list:
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")
    return 


    

if __name__ == "__main__":
    #print('A'+1)
    """
    raw_data_1 = read_data('/mnt/petrelfs/zhangdi1/ChemQA/data/train-00000-of-00002.parquet')
    raw_data_2 = read_data('/mnt/petrelfs/zhangdi1/ChemQA/data/train-00001-of-00002.parquet')
    raw_data = raw_data_1 + raw_data_2
    """
    raw_data = read_data('/mnt/petrelfs/zhangdi1/lijunxian/CheBI/test.csv')
    smiles_prompt_list = gen_smiles_dataset(raw_data, '/mnt/hwfile/ai4chem/share/cheBI/test/')
    caption_prompt_list = gen_caption_dataset(raw_data, '/mnt/hwfile/ai4chem/share/cheBI/test/')
    iupac_prompt_list = gen_iupac_dataset(raw_data, '/mnt/hwfile/ai4chem/share/cheBI/test/')

    total_list = smiles_prompt_list+caption_prompt_list+iupac_prompt_list
    write_total_data(total_list, '/mnt/petrelfs/zhangdi1/lijunxian/datagen/cheBI_test.jsonl')
    #convert_bytes_to_images(raw_data, '/mnt/hwfile/ai4chem/share/chemqa/val/', '/mnt/petrelfs/zhangdi1/lijunxian/datagen/chemqa_val.jsonl')