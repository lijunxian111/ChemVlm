# -*- coding: GBK -*-
import json
import os
from http import HTTPStatus
import numpy as np 
import os

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions

from tqdm import tqdm
import re

import dashscope
from dashscope import Generation
from dashscope import MultiModalConversation
from dashscope.api_entities.dashscope_response import Role

dashscope.api_key = 'sk-eea7c876461747c5a6eebe0531164767' #qwen API-KEY

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


def call_qwen(question: str):
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
        # stream=True,
        # incremental_output=True
    )
    full_content = ''
    # for response in responses:
    if response.status_code == HTTPStatus.OK:
        full_content = response.output.choices[0]['message']['content']

    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
    return full_content

def smiles_to_image(smiles_string, file_name):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles_string}")
    
    # Draw molecule and save as image file
    img = Draw.MolToImage(mol, size=(448, 448))
    img.save(file_name)
    return 

def read_and_create_mol2caption(path: str, restore_path: str):
    data_list = list()
    with open(path,'r', encoding='utf-8') as f:
        result = json.load(f)
    
    for idx, line in tqdm(enumerate(result)):
        smiles_start_index = line['question'].find('\n')
        smiles_str = line['question'][smiles_start_index+1:]
        
        q_id = f"mol2caption_{idx}"
        file_name = f"/path/to/chembench_mm/test/mol2caption_{idx}.png"
        try:
            smiles_to_image(smiles_str, file_name)
            choices = [k+'. '+v for k,v in line.items() if len(k) == 1]
            choices = ' '.join(choices)
            conv = [{'from': 'human', 'value': '<image>\n'+(line['question'].replace('SMILES',''))[0:smiles_start_index-1]+'in the image? ' + choices}, {'from':'gpt', 'value': line['answer']}]
            data_list.append({'id': q_id, 'images': [file_name], 'conversations': conv})
        except:
            pass

    writer = open(restore_path, 'w')
    for line in data_list:
        writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")

    return

def read_and_create_name_conversion(path: str, restore_path: str):

    template = "请帮我提取下面句子中的一个SMILES表达式。注意, 如果有SMILES表达式请仅回复这个表达式而不输出其他任何内容；如果没有请回复'-1'。{}"
    data_list = list()
    with open(path,'r', encoding='utf-8') as f:
        result = json.load(f)
    
    for idx, line in tqdm(enumerate(result)):
        smiles_start_index = line['question'].find(' C')
        if smiles_start_index == -1:
            continue
        smiles_str = line['question'][smiles_start_index:]

        q_id = f"name_conv_{idx}"
        file_name = f"/path/to/chembench_mm/test/name_conv_{idx}.png"
        try:
            smiles_to_image(smiles_str, file_name)
            choices = [k+'. '+v for k,v in line.items() if len(k) == 1]
            choices.sort()
            choices = ' '.join(choices)
            conv = [{'from': 'human', 'value': '<image>'+(line['question'].replace('SMILES','').replace(smiles_str, '').replace('?', ' in the image?')) + choices}, {'from':'gpt', 'value': line['answer']}]
            data_list.append({'id': q_id, 'images': [file_name], 'conversations': conv})
        except:
            pass

    writer = open(restore_path, 'w')
    for line in data_list:
        writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")

    return    

def read_and_create_retro(path: str, restore_path: str):

    #template = "请帮我提取下面句子中的一个SMILES表达式。注意, 如果有SMILES表达式请仅回复这个表达式而不输出其他任何内容；如果没有请回复'-1'。{}"
    data_list = list()
    with open(path,'r', encoding='utf-8') as f:
        result = json.load(f)
    
    for idx, line in tqdm(enumerate(result)):
        line_words = line['question'].split(' ')
        
        smiles_str = line_words[-2]
        q_id = f"retro_{idx}"
        file_name = f"/path/to/chembench_mm/test/retro_{idx}.png"
        
        smiles_to_image(smiles_str, file_name)
        choices = [k+'. '+v for k,v in line.items() if len(k) == 1]
        choices.sort()
        choices = ' '.join(choices)
        conv = [{'from': 'human', 'value': '<image>\n'+ line['question'].replace(smiles_str, 'the molecule in the image') + choices}, {'from':'gpt', 'value': line['answer']}]
        data_list.append({'id': q_id, 'images': [file_name], 'conversations': conv})
        #except:
        #pass

    writer = open(restore_path, 'w')
    for line in data_list:
        writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")

    return 

def read_and_create_solvent_pre(path: str, restore_path: str):
    
    data_list = list()
    with open(path,'r', encoding='utf-8') as f:
        result = json.load(f)
    
    for idx, line in tqdm(enumerate(result)):
        line_words = line['question'].split(' ')
        
        for word in line_words:
            if '>>' in word:
                reaction = word
                break
        
        
        reaction = reaction.replace('?','').replace(',','').replace('\'s','')
        reactants = (reaction.split('>>'))[0].split('.')
        product = (reaction.split('>>'))[1].replace('.','')

        try:
            q_id = f"solvent_{idx}"
            file_name = f"/path/to/chembench_mm/test/solvent_{idx}.png"
            draw_pic(reactants, [product], file_name)
            #smiles_to_image(smiles_str, file_name)
            choices = [k+'. '+v for k,v in line.items() if len(k) == 1]
            choices.sort()
            choices = ' '.join(choices)
            conv = [{'from': 'human', 'value': '<image>\n'+ line['question'].replace(reaction, 'the reaction in the image') + choices}, {'from':'gpt', 'value': line['answer']}]
            data_list.append({'id': q_id, 'images': [file_name], 'conversations': conv})
        except:
            pass

    writer = open(restore_path, 'w')
    for line in data_list:
        writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")

    return

def read_and_create_temperature_pre(path: str, restore_path: str):
    
    data_list = list()
    with open(path,'r', encoding='utf-8') as f:
        result = json.load(f)
    
    for idx, line in tqdm(enumerate(result)):
        line_words = line['question'].split(' ')
        
        for word in line_words:
            if '>>' in word or '>' in word:
                origin_reaction = word
                break
        
        
        reaction = origin_reaction.replace('?','').replace(',','').replace('\'s','').replace('>>','>')
        origin_reaction = origin_reaction.replace('?','').replace(',','').replace('\'s','')
        reactants = (reaction.split('>'))[0].split('.')
        product = (reaction.split('>'))[1].replace('.','')

        try:
            q_id = f"temperature_{idx}"
            file_name = f"/path/to/chembench_mm/test/temperature_{idx}.png"
            draw_pic(reactants, [product], file_name)
            #smiles_to_image(smiles_str, file_name)
            choices = [k+'. '+v for k,v in line.items() if len(k) == 1]
            choices.sort()
            choices = ' '.join(choices)
            conv = [{'from': 'human', 'value': '<image>'+ line['question'].replace(origin_reaction, 'in the image') + choices}, {'from':'gpt', 'value': line['answer']}]
            data_list.append({'id': q_id, 'images': [file_name], 'conversations': conv})
        except:
            pass

    writer = open(restore_path, 'w')
    for line in data_list:
        writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")

    return

def read_and_create_yield_pre(path: str, restore_path: str):
    
    data_list = list()
    with open(path,'r', encoding='utf-8') as f:
        result = json.load(f)
    
    for idx, line in tqdm(enumerate(result)):
        line_words = line['question'].split(' ')
        
        for word in line_words:
            if '>>' in word or '>' in word:
                origin_reaction = word
                break
        
        
        reaction = origin_reaction.replace('?','').replace(',','').replace('\'s','').replace('>>','>')
        origin_reaction = origin_reaction.replace('?','').replace(',','').replace('\'s','')
        reactants = (reaction.split('>'))[0].split('.')
        product = (reaction.split('>'))[1].replace('.','')

        try:
            q_id = f"yield_{idx}"
            file_name = f"/path/to/chembench_mm/test/yield_{idx}.png"
            draw_pic(reactants, [product], file_name)
            #smiles_to_image(smiles_str, file_name)
            choices = [k+'. '+v for k,v in line.items() if len(k) == 1]
            choices.sort()
            choices = ' '.join(choices)
            conv = [{'from': 'human', 'value': '<image>'+ line['question'].replace(origin_reaction, 'in the image') + choices}, {'from':'gpt', 'value': line['answer']}]
            data_list.append({'id': q_id, 'images': [file_name], 'conversations': conv})
        except:
            pass

    writer = open(restore_path, 'w')
    for line in data_list:
        writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")

    return

def read_and_create_mol2caption_for_training(path: str, restore_path: str):
    data_list = list()
    with open(path,'r', encoding='utf-8') as f:
        result = f.readlines()
    
    template = "请帮我提取下面句子中的一个SMILES表达式。注意, 你仅需要回答一个SMILES表达式而不需要输出别的任何内容。{}"
    for idx, line in tqdm(enumerate(result)):
        line = json.loads(line)
        q = template.format(line['instruction'])
        ans = call_qwen(q)
        q_id = f"train_{idx}"
        img_path = f"/path/to/chembench_mm/train/{q_id}.png"
        try:
            smiles_to_image(ans, img_path)
            #choices = [k+'. '+v for k,v in line.items() if len(k) == 1]
            #choices = ' '.join(choices)
            conv = [{'from': 'human', 'value': '<image>\n'+line['instruction'].replace('SMILES','').replace('molecule','molecule in the image').replace(ans, '').replace('\n',' ')}, {'from':'gpt', 'value': line['output']}]
            data_list.append({'id': q_id, 'images': [img_path], 'conversations': conv})
        except:
            pass
    
    writer = open(restore_path, 'w')
    for line in data_list:
        writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")

    return


def only_read_json(path: str, restore_path:str):
    data_list = list()
    with open(path,'r', encoding='utf-8') as f:
        result = f.readlines()
    
    for idx, line in enumerate(result):
        line = json.loads(line)
        """
        line['instruction'] = line['instruction'].replace('There is a single choice question about chemistry. Answer the question by replying A, B, C or D.\nQuestion: ', '')
        line['instruction'] = line['instruction'].replace('SMILES', '')
        if 'reaction' in line['instruction'] or 'Reaction' in line['instruction'] or 'synthesis' in line['instruction'] or 'creating' in line['instruction']:
            pass
        elif 'condition' in line['instruction'] or 'SELF' in line['instruction'] or 'IUPAC' in line['instruction']:
            pass
        elif 'Synthesize' in line['instruction'] or 'Generate' in line['instruction'] or 'Create' in line['instruction'] or 'Design' in line['instruction'] or 'create' in line['instruction'] or 'design' in line['instruction'] or 'yield' in line['instruction']:
            pass
        """
        if idx>=194 and idx<=223:
            pass
        else:
            data_list.append(line)
    
    writer = open(restore_path, 'w')
    for line in data_list:
        writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()   
    
    print("Finish!")

    return

if __name__ == "__main__":
    #only_read_json('../datagen/pattern_clean_training.jsonl', '../datagen/pattern_clean_training.jsonl')
    #read_and_create_mol2caption('../ChemBench/test/Mol2caption_benchmark.json', '../datagen/chembench_mol2caption.jsonl')
    #read_and_create_mol2caption_for_training('../datagen/pattern_clean.jsonl','../datagen/pattern_clean_training.jsonl')
    read_and_create_name_conversion('../ChemBench/test/Name_Conversion_benchmark.json', '../datagen/chembench_name_conv.jsonl')
    #read_and_create_retro('../ChemBench/test/Retrosynthesis_benchmark.json','../datagen/chembench_retro.jsonl')
    #read_and_create_solvent_pre('../ChemBench/test/Solvent_Prediction_benchmark.json', '../datagen/chembench_solvent.jsonl')
    #read_and_create_temperature_pre('../ChemBench/test/Temperature_Prediction_benchmark.json', '../datagen/chembench_temperature.jsonl')
    #read_and_create_yield_pre('../ChemBench/test/Yield_Prediction_benchmark.json','de')
