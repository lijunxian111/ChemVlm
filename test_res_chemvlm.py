# -*- coding: utf-8 -*-
import argparse
import json
import os
import random

import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess,
                                    find_closest_aspect_ratio, preprocess,
                                    preprocess_internlm_question, preprocess_mpt)
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from copy import deepcopy

ds_collections = {
    'smiles_ocr':{
        'root': '',
        'question': '/mnt/petrelfs/zhangdi1/lijunxian/datagen/mm_chem_ocr.jsonl.test.jsonl',
        'prompt': "请根据指令回答一个SMILES化学分子式。注意, 请只回答SMILES化学分子式而不输出其他内容",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
    'mm_gaokao_test_past': {
        'root': '/mnt/hwfile/ai4chem/share/data',
        'question': '/mnt/petrelfs/zhangdi1/lijunxian/datagen/mm_pure_fix.jsonl.test.jsonl',
        'prompt': "请判断这道是什么题目, 选择题请回答A, B, C或D; 填空题请按顺序依次填空; 主观题请回答问题并给出详细步骤",
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    }
}


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
            image_path_list = data_item['images']
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
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'] + self.prompt
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
            # history.append((question, response))
            print('question_id: ', question_id)
            print('response: ', response)
            print('annotation: ', annotation)
`           `
            outputs.append({
                'question_id': question_id,
                'text': response,
                'annotation': annotation,
                'model_id': model_id,
                'metadata': {}
            })

        print(f'Evaluating {ds_name} ...')
        results_file = 'gaokao_1k_' + model_id + '_' + ds_name +'.jsonl'
        results_file = os.path.join(args.out_dir, results_file)
        writer = open(results_file, 'w')
        for item in outputs:
            writer.write(json.dumps(item, ensure_ascii=False) + '\n')
        writer.close()
        print('Results saved to {}'.format(results_file))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--checkpoint', type=str, default='/mnt/petrelfs/zhangdi1/lijunxian/InternVL/pretrained/InternVL-Chat-V1-5')
    parser.add_argument('--checkpoint', type=str, default='/mnt/hwfile/ai4chem/CKPT/wxz/chemvl_2B_ft_7_3_0_merge')
    parser.add_argument('--datasets', type=str, default='smiles_ocr')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--conv-style', type=str, default='internlm2-chat')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results')
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
