from glob import glob
import json
import random

with open('./mm_chem_ocr.jsonl', 'w') as fm:
    for file in glob('/mnt/petrelfs/zhangdi1/lijunxian/datagen/cross_modal/*/data.jsonl'):
        if file == '/mnt/petrelfs/zhangdi1/lijunxian/datagen/cross_modal/qa/data.jsonl':
            continue
        if file == '/mnt/petrelfs/zhangdi1/lijunxian/datagen/cross_modal/latex/data.jsonl':
            print('hi!')
            with open(file) as f:
                lines = f.readlines()
                lines = [line.replace('"images": ["images/','"images": ["/mnt/petrelfs/zhangdi1/lijunxian/datagen/cross_modal/latex/images/') for line in lines]
                fm.write(''.join(lines))
                continue
        with open(file) as f:
            lines = f.readlines()
            fm.write(''.join(lines))

with open('./mm_chem_ocr.jsonl', 'r') as f2:
    lines = f2.readlines()
    print(len(lines))