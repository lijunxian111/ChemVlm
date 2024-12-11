import json
import random

base_path = './general_mm_data/llava/share_data_sft/playground/data/'

with open('./general_mm_pure.jsonl', 'w') as fm:
    with open('./general_text_pure.jsonl', 'w') as ft:
        with open('。/general_mm_data/llava/llava_v1_5_mix665k.jsonl') as f:
            lines = f.readlines()
            for line_txt in lines:
                line = json.loads(line_txt)
                if 'image' in line or 'images' in line:
                    if 'image' in line:
                        line['image'] = base_path + line['image']
                    if 'images' in line:
                        for i in range(len(line['images'])):
                            line['images'][i] = base_path + line['images'][i]
                    fm.write(json.dumps(line)+'\n')
                else:
                    ft.write(json.dumps(line)+'\n')

with open('./general_text_pure.jsonl', 'r') as f2:
    lines = f2.readlines()
    print(len(lines))
