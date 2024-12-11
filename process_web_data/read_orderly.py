# -*- coding: GBK -*-
import datasets as ds 
from datasets import concatenate_datasets
import json
import numpy as np
import os 

from typing import Dict, Sequence
import copy
import logging
from tqdm import tqdm
import re
import random

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions

"""
Please give me ten different expressions of template 
“In my opinion, analyzing the chemical reaction in the image, we can see the reactants are {} and {}. The product is {}. 
When represented with SMARTS, it should be {}."
Use as many different words, expressions and type of sentences as you can.
"""

"""
atomic_symbols = {
    "1": "H", "2": "He", "3": "Li", "4": "Be", "5": "B", "6": "C", "7": "N",
    "8": "O", "9": "F", "10": "Ne", "11": "Na", "12": "Mg", "13": "Al",
    "14": "Si", "15": "P", "16": "S", "17": "Cl", "18": "Ar", "19": "K",
    "20": "Ca", "21": "Sc", "22": "Ti", "23": "V", "24": "Cr", "25": "Mn",
    "26": "Fe", "27": "Co", "28": "Ni", "29": "Cu", "30": "Zn", "31": "Ga",
    "32": "Ge", "33": "As", "34": "Se", "35": "Br", "36": "Kr", "37": "Rb",
    "38": "Sr", "39": "Y", "40": "Zr", "41": "Nb", "42": "Mo", "43": "Tc",
    "44": "Ru", "45": "Rh", "46": "Pd", "47": "Ag", "48": "Cd", "49": "In",
    "50": "Sn", "51": "Sb", "52": "Te", "53": "I", "54": "Xe", "55": "Cs",
    "56": "Ba", "57": "La", "58": "Ce", "59": "Pr", "60": "Nd", "61": "Pm",
    "62": "Sm", "63": "Eu", "64": "Gd", "65": "Tb", "66": "Dy", "67": "Ho",
    "68": "Er", "69": "Tm", "70": "Yb", "71": "Lu", "72": "Hf", "73": "Ta",
    "74": "W", "75": "Re", "76": "Os", "77": "Ir", "78": "Pt", "79": "Au",
    "80": "Hg", "81": "Tl", "82": "Pb", "83": "Bi", "84": "Po", "85": "At",
    "86": "Rn"
}
"""

caption_templates = [
    "Could you explain the chemical reaction shown in the image?",
"Can you provide an explanation of the chemical reaction depicted in the image?",
"Would you mind describing the chemical reaction in this image for me?",
"Can you give me a detailed description of the chemical reaction illustrated in the image?",
"Could you walk me through the chemical reaction presented in the image?",
"Can you break down the chemical reaction shown in this image for me?",
"Would you explain the details of the chemical reaction depicted in the image?",
"Can you clarify how the chemical reaction in this image works?",
"Could you shed some light on the chemical reaction illustrated in the image?",
"Would you mind elaborating on the chemical reaction presented in this image?"
]

caption_answers = [
"From my perspective, by evaluating the chemical reaction in the image, we observe that the reactants are {} and {}. The resulting product is {}. In SMARTS format, it can be represented as {}.",
"In my view, an analysis of the chemical reaction shown in the image reveals that the reactants are {} and {}. The product formed is {}. When described using SMARTS, it should be {}.",
"In my opinion, examining the chemical reaction depicted in the image, we can determine that the reactants are {} and {}. The final product is {}. Represented in SMARTS, it would be {}.",
"As I see it, through scrutiny of the chemical reaction in the image, it's evident that the reactants are {} and {}. The outcome is {}. When translated to SMARTS, it is {}.",
"To me, the chemical reaction in the image shows that the reactants involved are {} and {}. The product yielded is {}. Using SMARTS notation, it should be expressed as {}.",
"I believe that, after analyzing the chemical reaction presented in the image, we can identify the reactants as {} and {}. The resulting compound is {}. In SMARTS terms, it would be {}.",
"From my analysis, the chemical reaction illustrated in the image involves {} and {} as reactants. The end product is {}. When converted to SMARTS, it reads as {}.",
"In my estimation, the image displays a chemical reaction where the reactants are {} and {}. The resultant product is {}. In SMARTS, this can be represented by {}.",
"It appears to me that the chemical reaction in the image involves {} and {} as reactants. The produced substance is {}. In the SMARTS format, this is represented as {}.",
"According to my analysis, the chemical reaction shown in the image has {} and {} as the reactants. The product of this reaction is {}. When denoted in SMARTS, it should be {}."
]

type_templates = [
"Would you describe the type of chemical reaction shown in the image?", 
"Can you explain the kind of chemical reaction depicted in the picture?", 
"Would you identify the category of the chemical reaction illustrated in the photo?", 
"Can you specify the type of chemical reaction presented in the diagram?", 
"Would you classify the chemical reaction shown in the image?", 
"Can you elaborate on the type of chemical reaction demonstrated in the picture?", 
"Would you determine the nature of the chemical reaction depicted in the illustration?", 
"Can you outline the type of chemical reaction featured in the visual?", 
"Would you clarify the kind of chemical reaction presented in the image?", 
"Can you pinpoint the type of chemical reaction shown in the photo?"
]

type_answers = [
"In my opinion, the type of chemical reaction depicted in this image is {}.",
"From what I can tell, this image illustrates a {} chemical reaction.",
"As I see it, the chemical reaction shown in this picture is a {}.",
"In my view, the reaction type represented in this image appears to be {}.",
"It seems to me that the chemical reaction in the picture is {}.",
"My assessment is that this image demonstrates a {} chemical reaction.",
"From my vantage point, the chemical reaction illustrated here is {}.",
"To my understanding, the type of reaction in this image is {}.",
"I believe the chemical reaction type shown in the picture is {}.",
"Based on my analysis, the image displays a {} chemical reaction."
]

def replace_atomic_number(match):
    atomic_number = match.group(1)
    return atomic_symbols.get(atomic_number, match.group(0))

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
    if len(raw_ans) == 4:
        return random.choice(template).format(raw_ans[0],raw_ans[1],raw_ans[2],raw_ans[3])
    elif len(raw_ans) == 3:
        ans_prompt = random.choice(template).format(raw_ans[0],"None", raw_ans[1],raw_ans[2])
        ans_prompt = ans_prompt.replace(" and None", "")
        return ans_prompt
    else:
        return random.choice(template).format(raw_ans)


def draw_pic(reactants: list, product: list, img_store_path: str):
    
    reactants_smarts = [Chem.MolToSmarts(Chem.MolFromSmiles(rea.replace(' ',''))) for rea in reactants]
    product_smarts = [Chem.MolToSmarts(Chem.MolFromSmiles(pro.replace(' ',''))) for pro in product]
    
    # 创建反应图
    rea_smarts = ".".join(reactants_smarts)
    total_smarts = rea_smarts + ">>" + product_smarts[0]
    
    reaction = rdChemReactions.ReactionFromSmarts(total_smarts, useSmiles=True)

    #accept_smarts = smarts = re.sub(r'\[#(\d+)\]', replace_atomic_number, total_smarts)
    accept_smarts = ".".join([rea.replace(' ','') for rea in reactants]) + ">>" + product[0].replace(' ','')
    #print(accept_smarts)
    # 绘制反应
    img = Draw.ReactionToImage(reaction)
    
    img_store_path = os.path.abspath(img_store_path) #规范path
    # 保存反应图
    if os.path.exists(img_store_path):
        pass
    else:
        img.save(img_store_path)
    return accept_smarts

def read_dataset(path: str):

    data = ds.load_from_disk(path)
    df = data.to_pandas()
    data = df.to_dict('records')
    pattern = r'\[([^\]]+)\]'  # 匹配以 [ 开头，] 结尾的内容，并且提取其中的内容
    #print(data[268])
    for i in tqdm(range(len(data)), desc='construct reactants and product'):
        reaction_list = data[i]['reaction'].split('.')
        rea_index = reaction_list[1].rfind(':')
        pro_index = reaction_list[2].rfind(':')
        reactants = reaction_list[1][rea_index+1:].split(', ')
        product = reaction_list[2][pro_index+1:]
        #draw_pic(reactants, [product])
        #break
        try:
            match = re.search(pattern, data[i]['reaction_type'])
            #print(match)
            rea_type = match.group(1)
            data[i]['reaction_type'] = rea_type
        except:
            data[i]['reaction_type'] = "Unclear"
        data[i]['reactants'] = reactants
        data[i]['product'] = [product]

    return data

def gen_caption_dataset(data,  img_store_path: str):
    prompt_list = list()
    for index, line in tqdm(enumerate(data), desc='gen caption'):
        q_id = 'orderly_caption' + str(index)
        prompt = get_prompt_from_templates(caption_templates)
        
        smarts = draw_pic(line['reactants'], line['product'], img_store_path + f"{index}.png") #画图并且返回代表反应的SMARTS，一个脚本只需要调用一次
        raw_answers = line['reactants'] + line['product'] + [smarts]
        ans_prompt = get_ans_from_templates(caption_answers, raw_answers)
        #q_type = line['question_type']
        
        imgs = []
     
        #image_np = cv2.imread(img_store_path + f"{index}.png")
        #将numpy矩阵保存为jpg格式的图片文件
        #cv2.imwrite(img_store_path + f"{index}.png", image_np) 只需要写入一次即可
        imgs.append(img_store_path + f"{index}.png")
        
        conversations = {'id': q_id, 'images': imgs, 'conversations':[{'from':'human', 'value': prompt}, {'from':'gpt', 'value': ans_prompt}]}
        

        prompt_list.append(conversations)
    
    return prompt_list

def gen_type_dataset(data, img_store_path: str):
    """
    参照gen_caption_dataset补全
    步骤：
    1. 自己生成问答模版,网页版gpt-3.5
    2. 生成问答句子
    3.生成对话
    """
    prompt_list = list()
    for index, line in tqdm(enumerate(data), desc='gen type'):
        q_id = 'orderly_type' + str(index)
        prompt = get_prompt_from_templates(type_templates)
        if "Unclear" in line['reaction_type']:
            continue
        else:

            smarts = draw_pic(line['reactants'], line['product'], img_store_path + f"{index}.png")
            raw_answers = line['reaction_type']
            ans_prompt = get_ans_from_templates(type_answers, raw_answers)

            imgs = []
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
    mode = ["train", "test"]
    for m in mode:
        raw_data = read_dataset(f'/orderly_data/orderly_{m}_final')
        caption_list = gen_caption_dataset(raw_data, f'/orderly_data/{m}/')
        rea_type_list = gen_type_dataset(raw_data, f'/orderly_data/{m}/')
        total_list = caption_list + rea_type_list
        write_total_data(total_list, f'/datagen/orderly_{m}.jsonl')

