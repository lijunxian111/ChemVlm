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

product_templates = [
    "We are currently observing a chemical reaction involving reactants as depicted in the provided diagram. {}. Could you please predict the products resulting from this reaction?",
    "Hey, we’ve got a chemical reaction going on with substances shown in the picture. {}. Can you take a guess at what the products might be?",
    "The reaction in question involves reactants illustrated in the accompanying figure. {}. Please provide a prediction of the products formed in this chemical process.",
    "Examine the chemical reaction diagram featuring the reactants. {}. Based on this information, predict the products that will be generated.",
    "Check out this awesome chemical reaction with the reactants shown in the image! {}. What do you think the products will be? I’m excited to hear your prediction!",
    "Given the chemical reaction with the reactants illustrated in the diagram, can you analyze and forecast the potential products of this reaction? Note that {}.",
    "There’s a reaction happening with chemicals shown in the picture. {}. What do you think will come out of this reaction?",
    "In light of the chemical reaction involving the reactants depicted in the diagram, could you provide an informed prediction regarding the products of this reaction? Note that {}.",
    "Look at the diagram showing the  reactants in this chemical reaction. {}. Based on what you see, what products do you think will result from this reaction?",
    "Imagine we’re running a chemical experiment with the reactants shown in the diagram. {}. What do you hypothesize will be the products of this reaction?",
]

product_answers = [
    "The products of this chemical reaction are as follows: {}.",
    "Alright, so the stuff that comes out of this reaction is {}.",
    "The resultant products of the chemical reaction are: {}.",
    "The outcome of this chemical reaction is: {}.",
    "Great news! The products of this reaction are {}! Isn’t that exciting?",
    "Upon analyzing the reaction, the resulting products are identified as: {}.",
    "Okay, the end result of this reaction is {}.",
    "The products of the chemical reaction have been determined to be: {}.",
    "So, after running the reaction, we find that the products are: {}.", 
    "If we consider the reaction, the resulting products would be: {}."
]

reactant_templates = [
    "We are currently analyzing a chemical reaction where the product is depicted in the provided diagram. {}{}Could you assist in predicting the other reactant involved in this reaction?",
    "So, we’ve got this reaction with the product shown in the picture. {}{}Can you figure out what the other reactant might be?",
    "The chemical reaction shown in the diagram has a product represented. {}{}Based on this information, please determine the additional reactant required for this reaction.",
    "Refer to the diagram displaying the product of the reaction. {}{}Could you predict what the other reactant for this reaction might be?",
    "Check out the product of this reaction in the image! {}{}Can you guess what the other reactant could be?",
    "Given the product shown in the reaction diagram, what do you deduce to be the other reactant necessary for completing this chemical process? Note that {}{}",
    "Here’s a picture of the product from this reaction. {}{}What do you think the other reactant is?",
    "In the reaction illustrated, the product is clearly shown in the diagram. {}{}Could you identify the other reactant involved in this chemical reaction?",
    "The diagram presents the product of the reaction. {}{}Based on this, can you predict what the other reactant in the reaction could be?",
    "Imagine we have a reaction with the product displayed in the image. {}{}What do you think the other reactant might be for this reaction?"
]

reactant_answers = [
    "The other reactant in this chemical reaction is {}.",
    "In this chemical reaction, the additional reactant is {}.",
    "The second reactant involved in this reaction is {}.",
    "Another substance required for this chemical reaction is {}.",
    "The alternative reactant for this reaction is {}.",
    "The supplementary reactant in this chemical process is {}.",
    "Thus, the remaining reactant for this reaction is {}.",
    "The other reactant participating in this chemical reaction is {}.",
    "For this chemical reaction, the additional reactant is {}.",
    "Hence, the other chemical reactant involved is {}."
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

def add_multiple_image_token(num, text):
    a = random.choice('01')
    if a == '0':
        text = '<image>\n' * num + text
    else:
        text = text + '\n<image>' * num
    return text

def get_prompt_from_templates(template):

    return add_image_token(random.choice(template))

def get_prompt_multiple_image_from_templates(template, num, text):

    return add_multiple_image_token(num, (random.choice(template)).format(text))

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

def draw_pic_for_smiles(smiles:list, img_store_paths: list):

    for index, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)

        # 生成分子图像
        img = Draw.MolToImage(mol, size=(448,448))

        s_path = os.path.abspath(img_store_paths[index])
        img.save(s_path)
    
    return 

def read_dataset(path: str):

    data = ds.load_from_disk(path)
    df = data.to_pandas()
    data = df.to_dict('records')
    pattern = r'\[([^\]]+)\]'  # 匹配以 [ 开头，] 结尾的内容，并且提取其中的内容
    #print(data[268])
    for i in tqdm(range(len(data)), desc='construct reactants and product'):
        #if i==0:
            #print(data[i])
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

def gen_type_dataset(data, img_store_path: str, test_or_not=False):
    """
    参照gen_caption_dataset补全
    步骤：
    1. 自己生成问答模版,网页版gpt-3.5
    2. 生成问答句子
    3.生成对话
    """
    prompt_list = list()
    cnt = 0
    cnt_neu = 0
    cnt_subs = 0
    for index, line in tqdm(enumerate(data), desc='gen type'):
        q_id = 'orderly_type' + str(index)
        prompt = get_prompt_from_templates(type_templates)
        if "Unclear" in line['reaction_type']:
            cnt += 1
            continue
        else:

            smarts = draw_pic(line['reactants'], line['product'], img_store_path + f"{index}.png")
            raw_answers = line['reaction_type']
            if "Nucleophilic" in raw_answers:
                cnt_neu += 1
                if cnt_neu > 5000:
                    continue
            if "Nucleophilic" not in raw_answers and "Substitution" in raw_answers:
                cnt_subs += 1
                if cnt_subs > 5000:
                    continue

            if test_or_not == True:
                ans_prompt = raw_answers
            else:
                ans_prompt = get_ans_from_templates(type_answers, raw_answers)


            imgs = []
            imgs.append(img_store_path + f"{index}.png")
            conversations = {'id': q_id, 'images': imgs, 'conversations':[{'from':'human', 'value': prompt}, {'from':'gpt', 'value': ans_prompt}]}
            prompt_list.append(conversations)

    print(f"Unclear Ones: {cnt}")
    return prompt_list

def gen_product_prediction_dataset(data, img_store_path: str, test_or_not=False):
    prompt_list = list()
    for index, line in tqdm(enumerate(data), desc='gen product prediction'):
        q_id = 'orderly_product' + str(index)
        prompt = get_prompt_multiple_image_from_templates(product_templates, len(line['reactants']), line['condition']+'. ')
        path_list = [img_store_path + f"pro_{index}_{i}.png" for i in range(len(line['reactants']))]
        
        draw_pic_for_smiles(line['reactants'], path_list)
        #smarts = draw_pic(line['reactants'], line['product'], img_store_path + f"{index}.png") #画图并且返回代表反应的SMARTS，一个脚本只需要调用一次
        #raw_answers = line['reactants'] + line['product'] + [smarts]
        if test_or_not == True:
            ans_prompt = line['product'][0]
        else:
            ans_prompt = get_ans_from_templates(product_answers, line['product'][0])
        #q_type = line['question_type']
        
        conversations = {'id': q_id, 'images': path_list, 'conversations':[{'from':'human', 'value': prompt}, {'from':'gpt', 'value': ans_prompt}]}
        

        prompt_list.append(conversations)

    return prompt_list

def gen_reactant_prediction_dataset(data, img_store_path: str, test_or_not=False):
    prompt_list = list()
    for index, line in tqdm(enumerate(data), desc='gen reactant prediction'):
        q_id = 'orderly_reactant' + str(index)
        prompt = '<image>\n'+ random.choice(reactant_templates)
        if len(line['reactants']) == 1:
            prompt = prompt.format('', line['condition']+'. ')
            prompt = prompt.replace('the other', 'the')

        else:
            prompt = prompt.format('One reactant is also in the image\n<image>\n', line['condition']+'. ')

        path_list = [img_store_path + f"rea_{index}_{i}.png" for i in range(len(line['reactants']))]
        
        smiles_list = []
        smiles_list.append(line['product'][0])
        if len(line['reactants'])==2:
            chosen_smiles = random.choice(line['reactants'])
            smiles_list.append(chosen_smiles)

        new_rea_lst = line['reactants']
        if len(line['reactants'])==2:
            new_rea_lst.remove(chosen_smiles)
        draw_pic_for_smiles(smiles_list, path_list)
        #smarts = draw_pic(line['reactants'], line['product'], img_store_path + f"{index}.png") #画图并且返回代表反应的SMARTS，一个脚本只需要调用一次
        #raw_answers = line['reactants'] + line['product'] + [smarts]
        if test_or_not == True:
            ans_prompt = new_rea_lst[0]
        else:
            ans_prompt = get_ans_from_templates(reactant_answers, new_rea_lst[0])
        #q_type = line['question_type']
        
        conversations = {'id': q_id, 'images': path_list, 'conversations':[{'from':'human', 'value': prompt}, {'from':'gpt', 'value': ans_prompt}]}


        prompt_list.append(conversations)

    return prompt_list

def remove_elements(list1, list2):
      # 将 list2 转换为集合
    return [item for item in list1 if item not in list2]


def write_total_data(prompt_list: list, file_path: str):
    writer = open(file_path, 'w')
    for item in prompt_list:
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")
    return 


if __name__ == "__main__":
    
    mode = "test" #["train", "test"]

    raw_data = read_dataset(f'/mnt/hwfile/ai4chem/share/orderly_data/orderly_{mode}_final')
    random.shuffle(raw_data)
    if mode == "train":
        fill_in_data_rea = raw_data[0:50000] #填空型数据
        #des_data = remove_elements(raw_data, fill_in_data_rea) #描述型数据
        fill_in_data_pro = raw_data[50001:100000]
        des_data = raw_data[100001:200000]
        type_data = raw_data[100001:]
        test_or_not = False
    else:
        fill_in_data_rea = raw_data[0:5000] #填空型数据
        #des_data = remove_elements(raw_data, fill_in_data_rea) #描述型数据
        fill_in_data_pro = raw_data[5001:10000]
        type_data = raw_data
        des_data = raw_data[10001:]
        test_or_not = True
    #des_data = raw_data
    
    
    product_prediction_list = gen_product_prediction_dataset(fill_in_data_pro, f'/mnt/hwfile/ai4chem/share/orderly_data/{mode}/', test_or_not)
    reactant_prediction_list = gen_reactant_prediction_dataset(fill_in_data_rea, f'/mnt/hwfile/ai4chem/share/orderly_data/{mode}/', test_or_not)

    caption_list = gen_caption_dataset(des_data, f'/mnt/hwfile/ai4chem/share/orderly_data/{mode}/')
    rea_type_list = gen_type_dataset(type_data, f'/mnt/hwfile/ai4chem/share/orderly_data/{mode}/', test_or_not)

    total_list = reactant_prediction_list + product_prediction_list + caption_list + rea_type_list
    

    write_total_data(total_list, f'/mnt/petrelfs/zhangdi1/lijunxian/datagen/orderly_{mode}.jsonl')

    
    

    

