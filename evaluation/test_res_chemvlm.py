# -*- coding: utf-8 -*-
import argparse
import json
import os
import random

import re

import torch

import sys
sys.path.append('/mnt/hwfile/ai4chem/hao/Chemvlm_work/InternVL/internvl_chat/')

from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset_old import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess,
                                    find_closest_aspect_ratio, preprocess,
                                    preprocess_internlm_question, 
                                    preprocess_mpt)
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer
from copy import deepcopy
from typing import Dict

"""
ds_collections = {
    'smiles_ocr':{
        'root': '',
        'question': './datagen/mm_chem_ocr.jsonl.test.jsonl',
        'prompt': "请根据指令回答一个SMILES化学分子式, 请只回答SMILES化学分子式而不输出其他内容",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'mm_gaokao_test_past': {
        'root': '/mnt/hwfile/ai4chem/share/data',
        'question': './datagen/mm_pure_fix.jsonl.test.jsonl',
        #'prompt': "请判断这道是什么题目并回答, 选择题请只回答A, B, C或D; 填空题请按顺序依次填空,并只回答填入的内容; 主观题请回答问题并给出详细步骤",
        'prompt': "请正确回答这道题目, 选择题请只回答A, B, C或D; 填空题请按顺序依次填空,并只回答填入的内容",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'CMMU':{
        'root': '',
        'question': './datagen/CMMU_test_no_multiple.jsonl',
        'prompt': "请正确回答这道题目, 选择题请只回答A, B, C或D; 填空题请按顺序依次填空,并只回答填入的内容",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'SciQA':{
        'root': '',
        'question': './SciQA/sciqa_test.jsonl',
        'prompt': "Please answer A, B, C or D according to the question. Your answer should follow the format: ```{'answer': []}```. '[]' means a capital letter, A, B, C or D. Remember, you shouldn't return any other words.",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'chirality': {
       'root': '',
       'question': '/mnt/hwfile/ai4chem/hao/data_processing/chirality_mol_folder/combined_mol_data_easy_test.jsonl',
       'prompt': "Your answer should only contain a yes or no and a number. For example,yes,2 or no,0.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    }
}
"""


ds_collections = {
    'smiles_ocr':{
        'root': '',
        'question': './datagen/mm_chem_ocr.jsonl.test.jsonl',
        'prompt': "\n请根据指令回答一个SMILES化学表达式, 请只回答SMILES化学表达式而不输出其他内容",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'mm_gaokao_test_past': {
        'root': '/mnt/hwfile/ai4chem/share/data/',
        'question': './datagen/mm_pure_fix.jsonl.test.jsonl',
        #'prompt': "请判断这道是什么题目并回答, 选择题请只回答A, B, C或D; 填空题请按顺序依次填空,并只回答填入的内容; 主观题请回答问题并给出详细步骤",
        'prompt': "\n请正确回答这道题目, 选择题请只回答一个字母A, B, C或D; 填空题请按顺序依次填空,并只回答填入的内容",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'CMMU':{
        'root': '',
        'question': './datagen/CMMU_test_no_multiple.jsonl',
        'prompt': "\n请正确回答这道题目, 选择题请只回答一个字母A, B, C或D; 填空题请按顺序依次填空, 并只回答填入的内容",
        #'prompt': "这是一道多项选择题。请只回答你认为的多个正确选项, 下面是一个示例, ```烧杯中盛有$${CuCl_{2}}$$和$${HCl}$$的混合溶液$${100g}$$，向其中滴加$${10\\%}$$的$${NaOH}$$溶液，烧杯中溶液的质量与滴加溶液的质量关系如图所示.下列说法正确的是(　　)A.$${ab}$$段反应产生蓝色沉淀 B.$${bc}$$段溶液增加$${70.2g}$$ C.c点对应的溶质质量分数为$${4.9\\%}$$ D.d点溶液显碱性。请注意有多个答案是正确的```。对这个示例,你的回答应该是'BD'",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'CMMU_bio':{
        'root': '',
        'question': './datagen/CMMU_test_biology.jsonl',
        'prompt': "\n请正确回答这道题目, 选择题请只回答一个字母A, B, C或D; 填空题请按顺序依次填空, 并只回答填入的内容",
        #'prompt': "这是一道多项选择题。请只回答你认为的多个正确选项, 下面是一个示例, ```烧杯中盛有$${CuCl_{2}}$$和$${HCl}$$的混合溶液$${100g}$$，向其中滴加$${10\\%}$$的$${NaOH}$$溶液，烧杯中溶液的质量与滴加溶液的质量关系如图所示.下列说法正确的是(　　)A.$${ab}$$段反应产生蓝色沉淀 B.$${bc}$$段溶液增加$${70.2g}$$ C.c点对应的溶质质量分数为$${4.9\\%}$$ D.d点溶液显碱性。请注意有多个答案是正确的```。对这个示例,你的回答应该是'BD'",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'CMMU_math':{
        'root': '',
        'question': './datagen/CMMU_test_math.jsonl',
        'prompt': "\n请正确回答这道题目, 选择题请只回答一个字母A, B, C或D; 填空题请按顺序依次填空, 并只回答填入的内容",
        #'prompt': "这是一道多项选择题。请只回答你认为的多个正确选项, 下面是一个示例, ```烧杯中盛有$${CuCl_{2}}$$和$${HCl}$$的混合溶液$${100g}$$，向其中滴加$${10\\%}$$的$${NaOH}$$溶液，烧杯中溶液的质量与滴加溶液的质量关系如图所示.下列说法正确的是(　　)A.$${ab}$$段反应产生蓝色沉淀 B.$${bc}$$段溶液增加$${70.2g}$$ C.c点对应的溶质质量分数为$${4.9\\%}$$ D.d点溶液显碱性。请注意有多个答案是正确的```。对这个示例,你的回答应该是'BD'",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'CMMU_his':{
        'root': '',
        'question': './datagen/CMMU_test_history.jsonl',
        'prompt': "\n请正确回答这道题目, 选择题请只回答一个字母A, B, C或D; 填空题请按顺序依次填空, 并只回答填入的内容",
        #'prompt': "这是一道多项选择题。请只回答你认为的多个正确选项, 下面是一个示例, ```烧杯中盛有$${CuCl_{2}}$$和$${HCl}$$的混合溶液$${100g}$$，向其中滴加$${10\\%}$$的$${NaOH}$$溶液，烧杯中溶液的质量与滴加溶液的质量关系如图所示.下列说法正确的是(　　)A.$${ab}$$段反应产生蓝色沉淀 B.$${bc}$$段溶液增加$${70.2g}$$ C.c点对应的溶质质量分数为$${4.9\\%}$$ D.d点溶液显碱性。请注意有多个答案是正确的```。对这个示例,你的回答应该是'BD'",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'CMMU_phy':{
        'root': '',
        'question': './datagen/CMMU_test_physics.jsonl',
        'prompt': "\n请正确回答这道题目, 选择题请只回答一个字母A, B, C或D; 填空题请按顺序依次填空, 并只回答填入的内容",
        #'prompt': "这是一道多项选择题。请只回答你认为的多个正确选项, 下面是一个示例, ```烧杯中盛有$${CuCl_{2}}$$和$${HCl}$$的混合溶液$${100g}$$，向其中滴加$${10\\%}$$的$${NaOH}$$溶液，烧杯中溶液的质量与滴加溶液的质量关系如图所示.下列说法正确的是(　　)A.$${ab}$$段反应产生蓝色沉淀 B.$${bc}$$段溶液增加$${70.2g}$$ C.c点对应的溶质质量分数为$${4.9\\%}$$ D.d点溶液显碱性。请注意有多个答案是正确的```。对这个示例,你的回答应该是'BD'",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'CMMU_geo':{
        'root': '',
        'question': './datagen/CMMU_test_geography.jsonl',
        'prompt': "\n请正确回答这道题目, 选择题请只回答一个字母A, B, C或D; 填空题请按顺序依次填空, 并只回答填入的内容",
        #'prompt': "这是一道多项选择题。请只回答你认为的多个正确选项, 下面是一个示例, ```烧杯中盛有$${CuCl_{2}}$$和$${HCl}$$的混合溶液$${100g}$$，向其中滴加$${10\\%}$$的$${NaOH}$$溶液，烧杯中溶液的质量与滴加溶液的质量关系如图所示.下列说法正确的是(　　)A.$${ab}$$段反应产生蓝色沉淀 B.$${bc}$$段溶液增加$${70.2g}$$ C.c点对应的溶质质量分数为$${4.9\\%}$$ D.d点溶液显碱性。请注意有多个答案是正确的```。对这个示例,你的回答应该是'BD'",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'CMMU_pol':{
        'root': '',
        'question': './datagen/CMMU_test_politics.jsonl',
        'prompt': "\n请正确回答这道题目, 选择题请只回答一个字母A, B, C或D; 填空题请按顺序依次填空, 并只回答填入的内容",
        #'prompt': "这是一道多项选择题。请只回答你认为的多个正确选项, 下面是一个示例, ```烧杯中盛有$${CuCl_{2}}$$和$${HCl}$$的混合溶液$${100g}$$，向其中滴加$${10\\%}$$的$${NaOH}$$溶液，烧杯中溶液的质量与滴加溶液的质量关系如图所示.下列说法正确的是(　　)A.$${ab}$$段反应产生蓝色沉淀 B.$${bc}$$段溶液增加$${70.2g}$$ C.c点对应的溶质质量分数为$${4.9\\%}$$ D.d点溶液显碱性。请注意有多个答案是正确的```。对这个示例,你的回答应该是'BD'",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'SciQA':{
        'root': '',
        'question': './SciQA/sciqa_test.jsonl',
        'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'SciQA_all':{
        'root': '',
        'question': './datagen/sciqa_test_all.jsonl',
        'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'Chembench_mol2cap': {
       'root': '',
       'question': './datagen/chembench_mol2caption.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_property': {
       'root': '',
       'question': './datagen/chembench_property.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_name_conv': {
       'root': '',
       'question': './datagen/chembench_name_conv.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_retro': {
       'root': '',
       'question': './datagen/chembench_retro.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_temperature': {
       'root': '',
       'question': './datagen/chembench_temperature.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_solvent': {
       'root': '',
       'question': './datagen/chembench_solvent.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chembench_yield': {
       'root': '',
       'question': './datagen/chembench_yield.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 1000,
       'min_new_tokens': 1,
    },
    'Chebi_caption':{
       'root': '',
       'question': './datagen/cheBI_caption_eng_test.jsonl',
       'prompt': "Please follow the question and give your answer. Answer a few sentences in English. Here is one example answer: \n'''It seems to me the molecule is an {} in which {}. It has a role as an {}. It is an {}, {}, {}. It derives from a {}.'''\n. Follow this format.",
       'max_new_tokens': 200,
       'min_new_tokens': 1,
    },
    'orderly_type':{
       'root': '',
       'question': './datagen/orderly_test_type.jsonl',
       'prompt': "Please follow the question and give your answer. Choose one from '''{}'''. Remember that only return several words of the reaction type and do not return explanations.",
       'max_new_tokens': 200,
       'min_new_tokens': 1,
    },
    'pistachio_type_sample':{
       'root': '',
       'question': './datagen/pistachio_mm_test_1000.jsonl',
       'prompt': "Please follow the question and give your answer. Choose one from '''{}'''. Remember that only return the reaction type and do not return any other words",
       'max_new_tokens': 200,
       'min_new_tokens': 1,
    },
    'math_verse':{
       'root': '',
       'question': './datagen/math_verse_test.jsonl',
       'prompt': "Your answer format should be ```{Answer: }```",
       'max_new_tokens': 100,
       'min_new_tokens': 1,
    },
    'mmtbench':{
       'root': '',
       'question': './datagen/mmtbench_val.jsonl',
       'prompt': "Please answer A, B, C or D according to the question. You should only answer a capital letter.",
       'max_new_tokens': 100,
       'min_new_tokens': 1,
    }
}


with open('./eval_results/gpt-4o-pistachio_type_sample.jsonl', 'r', encoding='utf8') as f:
    type_data = f.readlines()

f.close()
all_types = [json.loads(item)['annotation'] for item in type_data]
#all_types.remove('sub')
all_types = json.dumps(list(set(all_types)))
ds_collections['pistachio_type_sample']['prompt'] = ds_collections['pistachio_type_sample']['prompt'].format(all_types)

class GaoKaoDataset(torch.utils.data.Dataset):

    def __init__(self, root, data, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.root = root
        self.data = open(data).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.tcs_loader = None
        if args.conv_style == 'internlm2-chat':
            self.preprocess_function = preprocess_internlm_question
        else:
            raise ValueError("wrong template")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = json.loads(self.data[idx].strip())
        # question, question_id, annotation =  data_item[
        #     'text'], data_item['id'], data_item.get('answer', None)
        question_id = data_item['id']
        annotation = data_item['conversations'][1]["value"] #TODO
        if 'image' in data_item:
            image_path_list = [data_item['image']]
        elif 'images' in data_item:
            image_path_list = [item.replace('CMMUval','CMMU/val') for item in data_item['images']]
        else:
            image_path_list = []
        
        
        images = []
        for image_path in image_path_list:
            if image_path.startswith('s3://'):
                image_path = self.root + image_path
            else:
                image_path = os.path.join(self.root, image_path)
            if self.tcs_loader is not None:
                image = self.tcs_loader(image_path)
            else:
                image = Image.open(image_path).convert('RGB')

            if self.dynamic_image_size:
                images.extend(dynamic_preprocess(image, max_num=self.max_num,
                                                 image_size=self.input_size, use_thumbnail=self.use_thumbnail))
            else:
                images.append(image)
        
        if images:
            pixel_values = [self.transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
        
        if not images:
            pixel_values = None
            # images.append(Image.new('RGB', (224, 224), (255, 255, 255)))
          
        
        # question = question + self.prompt
        # print(pixel_values)
        # print(data_item['conversations'])
        data_item['conversations'][0]['value'] = self.prompt + data_item['conversations'][0]['value']
        ret = self.preprocess_function(args.conv_style, deepcopy(data_item['conversations']),
                                tokenizer, model.num_image_token,
                                group_by_length=False)

        return question_id, ret['input_ids'], ret['attention_mask'], pixel_values, annotation


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = GaoKaoDataset(
            root=ds_collections[ds_name]['root'],
            data=ds_collections[ds_name]['question'],
            prompt=ds_collections[ds_name]['prompt'],
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )

        outputs = []
        for _, (question_id, input_ids, attention_mask, pixel_values, annotation) in tqdm(enumerate(dataset)):
            #if "caption" not in question_id:
                #continue
            
            if pixel_values is not None:
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(['<|im_end|>'])[0]]
            )
            
            generation_output = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                **generation_config
            )
            #print(generation_output)
            response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            response = response.split('<|im_end|>')[0].strip()  # for InternLM2

            #if re.search(r'[\u4e00-\u9fff]', response) is not None:
                #continue
            # history.append((question, response))
            print('question_id: ', question_id)
            print('response: ', response)
            print('annotation: ', annotation)

            outputs.append({
                'question_id': question_id,
                'text': response,
                'annotation': annotation,
                'model_id': model_id,
                'metadata': {}
            })

        print(f'Evaluating {ds_name} ...')
        results_file = args.task + '_' + model_id + '_' + ds_name +'.jsonl'
        results_file = os.path.join(args.out_dir, results_file)
        writer = open(results_file, 'w')
        for item in outputs:
            writer.write(json.dumps(item, ensure_ascii=False) + '\n')
        writer.close()
        print('Results saved to {}'.format(results_file))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='pistachio_type_sample')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--conv-style', type=str, default='internlm2-chat')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--task', type=str, default='reaction')
    parser.add_argument('--out-dir', type=str, default='./eval_results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).cuda().eval()
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    model_id = '_'.join(args.checkpoint.split('/')[-2:])
    evaluate_chat_model()
