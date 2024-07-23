# -*- coding: GBK -*-
import json
import numpy as np 

from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel

import torch
from scipy.spatial.distance import cosine

def move_vector_to_gpu(encoded_dict):
    if torch.cuda.is_available():
        encoded_dict = {key: value.to('cuda') for key, value in encoded_dict.items()}
    
    return encoded_dict

def cal_similarity(model, sentence1: str, sentence2: str):
    """
    encoded_input_1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors='pt')
    encoded_input_2 = tokenizer(sentence1, padding=True, truncation=True, return_tensors='pt')

    encoded_input_1 = move_vector_to_gpu(encoded_input_1)
    encoded_input_2 = move_vector_to_gpu(encoded_input_2)
    """

    with torch.no_grad():
        model_output_1 = (model.encode(sentence1, max_length=512)['dense_vecs'])
        model_output_2 = (model.encode(sentence2, max_length=512)['dense_vecs'])

    #sentence_embeddings_1 = mean_pooling(model_output_1, encoded_input_1['attention_mask']).cpu().numpy()
    #sentence_embeddings_2 = mean_pooling(model_output_2, encoded_input_1['attention_mask']).cpu().numpy()

    #return 1 - cosine(sentence_embeddings_1, sentence_embeddings_2)
    return (model_output_1 @ model_output_2.T)

def cal_sim_all_items(path_1, path_2):
    with open(path_1, 'r') as f:
        data_1 = f.readlines()
    with open(path_2, 'r') as fp:
        data_2 = fp.readlines()
    
    #tokenizer = AutoTokenizer.from_pretrained('sentence-transformers')
    model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)

    for line in data_1:
        total_sim = 0.
        cnt = 0.

        for line_2 in data_2:
            s1 = json.loads(line)
            s1 = s1['conversations'][0]['value']
            s2 = json.loads(line_2)
            s2 = s2['conversations'][0]['value']
            sim = cal_similarity(model, s1, s2)
            total_sim += sim
            cnt += 1 
        print(f"Line similarity: {total_sim/cnt}")
    
    print(f"File similarity: {total_sim/cnt}")

if __name__ == "__main__":

    cal_sim_all_items('/mnt/petrelfs/zhangdi1/lijunxian/SciQA/sciqa_test.jsonl', '/mnt/petrelfs/zhangdi1/lijunxian/datagen/mm_pure_fix.jsonl.test.jsonl')
