import json
import random

base_path = './general_chinese_mm/images/'

with open('./general_mm_pure_chinese.jsonl', 'w') as fm:
    with open('./general_chinese_mm/instruction_tuning.jsonl') as f:
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
                pass

with open('./general_mm_pure_chinese.jsonl', 'r') as f2:
    lines = f2.readlines()
    print(len(lines))
