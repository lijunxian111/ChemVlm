import matplotlib.pyplot as plt 
import numpy
import json
import re
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from nltk.translate.bleu_score import SmoothingFunction

from transformers import AutoTokenizer

#from bleu import multi_list_bleu, list_bleu

"""
def bleu_score_fn(text, gt):
    reference = [text.split(' ')]

    candidate = gt.split(' ')

    smooth = SmoothingFunction() # 定义平滑函数对象

    score = sentence_bleu(reference, candidate, weight=(0.25,0.25, 0.25, 0.25), smoothing_function=smooth.method1)

    return score
"""

def call_bleu_scores(path: str, tokenizer_path: str):
    with open(path, 'r') as f:
        json_data = f.readlines()
    
    smooth = SmoothingFunction()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
    candidates = list()
    references = list()

    cnt = 0.
    total_cnt = 0.

    for index, line in enumerate(json_data):
        line = json.loads(line)
        
        gt_tokens = line['annotation']
        gt_tokens = tokenizer.tokenize(gt_tokens)
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))
        
        try:
            out_tokens = tokenizer.tokenize(line['text'])
        except:
            out_tokens = tokenizer.tokenize(line['text'][0]['text']) #for json output
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))
        

        references.append([gt_tokens])
        candidates.append(out_tokens)
        #bleu_score = list_bleu([line['annotation']], [line['text']])

        #cnt += bleu_score
        #total_cnt += 1.0
    
    bleu2 = corpus_bleu(references, candidates, weights=(.5,.5), smoothing_function=smooth.method1)
    bleu4 = corpus_bleu(references, candidates, weights=(.25,.25,.25,.25), smoothing_function=smooth.method1)

    #print(f"Average Bleu: {cnt/total_cnt}")
    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    
    return

def get_type_scores(path: str):
    with open(path, 'r') as f:
        json_data = f.readlines()
    
    cnt = 0.
    total_cnt = 0.
    for index, line in enumerate(json_data):
        line = json.loads(line)
        if line['annotation'] in line['text']:
        #if line['annotation'].lower() in line['text'].lower():
            #if line['annotation'].lower() == 'substitution' and 'nucleophilic' in line['text'].lower():
                #pass
            #else:
                cnt += 1
        total_cnt+=1
    print(cnt)
    print(total_cnt)
    print(cnt/total_cnt)
    return

def cal_choice_scores(path: str):
    with open(path, 'r') as f:
        json_data = f.readlines()
    
    cnt = 0.
    total_cnt = 0.
    for index, line in enumerate(json_data):
        line = json.loads(line)
        if line['text'][0] == line['annotation']:
            cnt +=1 
        #elif type(line['text'])==type("a") and (re.findall("[A-Z]", line['text']))[0] == line['annotation']:
            #cnt += 1
        elif type(line['text'])==type("a") and "".join(re.findall("[A-Z]", line['text'])) == line['annotation']:
            cnt += 1
        
        elif type(line['text']) == type(['a']):
            gpt_chose = line['text'][0]['text']
            if gpt_chose == line['annotation']:
                cnt += 1
            elif "".join(re.findall("[A-Z]", gpt_chose)) == line['annotation']:
                cnt += 1
        total_cnt+=1
    print(cnt)
    print(total_cnt)
    print(cnt/total_cnt)
    return

def get_visualization(path: str):
    with open(path, 'r') as f:
        json_data = f.readlines()
    
    line_0 = json.loads(json_data[799])
    q = line_0['conversations'][0]['value']
    print(q)
    print(line_0['conversations'][1]['value'])
    img = '/mnt/hwfile/ai4chem/share/data/' + line_0['images'][0]
    pic = Image.open(img)
    pic.save('/mnt/petrelfs/zhangdi1/lijunxian/sample2.png')

    print(q)

def get_pure_eng_text(path: str, store_path: str):

    store_list = []
    with open(path, 'r') as f:
        json_data = f.readlines()
    
    for line in json_data:
        line = json.loads(line)
    
        if 'caption' not in line['id'] or re.search(r'[\u4e00-\u9fff]', line['conversations'][0]['value']) is not None:
            continue
        
        store_list.append(line)

    with open(store_path, 'w') as writer:
        for item in store_list:
            writer.write(json.dumps(item, ensure_ascii=False)+'\n')    

    f.close()
    writer.close()        


def smiles_to_image(smiles_string, file_name):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles_string}")
    
    # Draw molecule and save as image file
    img = Draw.MolToImage(mol, size=(448, 448))
    img.save(file_name)
    return

def read_lines(path: str):
    with open(path, 'r') as f:
        json_data = f.readlines()
    
    new_lst = []
    total_cnt = 0
    cnt = 0
    rea_dict = {}
    for line in json_data:
        line = json.loads(line)
        if 'type' in line['id']:
            if "Nucleophilic" in line['conversations'][1]['value'] and ('Substitution' in line['conversations'][1]['value'] or 'substitution' in line['conversations'][1]['value']):
                cnt += 1
                if cnt > 2500:
                    continue
        
        new_lst.append(line)
    
    with open(path, 'w') as writer:
        for item in new_lst:
            writer.write(json.dumps(item, ensure_ascii=False)+'\n')    

    f.close()
    writer.close()    
            
    
    print(cnt)
    print(total_cnt)





if __name__ == "__main__":
    """
    with open('/mnt/petrelfs/zhangdi1/lijunxian/eval_results/qwen-vl_dpo-mmstar.jsonl', 'r') as f:
        data = f.readlines()
    new_data = []
    for line in data:
        line = json.loads(line)
        if len(line['generated']) > 1:
            pass
        else:
            new_data.append(line)
    with open('/mnt/petrelfs/zhangdi1/lijunxian/eval_results/qwen-vl_dpo-mmstar.jsonl', 'w') as writer:
        for line in new_data:
            writer.write(json.dumps(line, ensure_ascii=False)+'\n')
    
    f.close()
    writer.close()
    """
    
    #print(json.loads(data[988]))
    #read_lines_2('/mnt/petrelfs/zhangdi1/lijunxian/eval_results/qwen-vl_mmstar-dpo-critic.jsonl')
    #smiles_to_image('CCOC(=O)C','sample.png')
    
    #print(len(data))
    #cal_choice_scores('/mnt/petrelfs/zhangdi1/lijunxian/eval_results/result_chemvlm26B_1120__SciQA.jsonl')
    #read_lines('/mnt/petrelfs/zhangdi1/lijunxian/datagen/CMMU_test_no_multiple.jsonl')
    #get_visualization('/mnt/petrelfs/zhangdi1/lijunxian/datagen/mm_pure_fix.jsonl.test.jsonl')
    #read_lines('/mnt/petrelfs/zhangdi1/lijunxian/datagen/orderly_train.jsonl')
    #call_bleu_scores('/mnt/petrelfs/zhangdi1/lijunxian/eval_results/result_chemvlm26B_1110__Chebi_caption.jsonl','/mnt/hwfile/ai4chem/share/InternVL2-26B')
    #get_pure_eng_text('/mnt/petrelfs/zhangdi1/lijunxian/datagen/cheBI_test.jsonl','/mnt/petrelfs/zhangdi1/lijunxian/datagen/cheBI_caption_eng_test.jsonl')
    #call_bleu_scores('/mnt/petrelfs/zhangdi1/lijunxian/eval_results/result_chemvlm26B_1110__Chebi_caption.jsonl','google-bert/bert-base-uncased')
    get_type_scores('/mnt/petrelfs/zhangdi1/lijunxian/eval_results/reaction_share_InternVL2-26B_pistachio_type_sample.jsonl')