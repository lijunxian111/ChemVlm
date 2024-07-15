import json
import random

base_path = '/mnt/petrelfs/zhangdi1/lijunxian/datagen/cross_modal/qa/'

with open('./general_mm_exam.jsonl', 'w') as fm:
    with open('/mnt/petrelfs/zhangdi1/lijunxian/datagen/cross_modal/qa/data.jsonl') as f:
        lines = f.readlines()
        for line_txt in lines:
            line = json.loads(line_txt)
            if 'image' in line or 'images' in line:
                if 'image' in line:
                    line['image'] = base_path + line['image']
                if 'images' in line:
                    for i in range(len(line['images'])):
                        line['images'][i] = base_path + line['images'][i]
                    if len(line['images']) != line['conversations'][0]['value'].count('<image>'):
                        continue
                fm.write(json.dumps(line)+'\n')

with open('./general_mm_exam.jsonl', 'r') as f2:
    lines = f2.readlines()
    print(len(lines))