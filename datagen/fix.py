from glob import glob
import json


# def func(lines):
#     for line in lines:
#         a = json.loads(line)
#         count = line.count('<image>')
#         lens = 0
#         if 'image' in a:
#             lens += 1
#         if 'images' in a:
#             lens += len(a['images'])
#         if count != lens:
#             print(file)
#             return

# for file in glob('./general_mm_exam.jsonl'):
#     # print(file)
#     with open(file) as f:
#         lines = f.readlines()
#         func(lines)


import json
import random

base_path = '/mnt/petrelfs/zhangdi1/lijunxian/datagen/cross_modal/qa/'

with open('./ultrachat_200k_fix.jsonl', 'w') as fm:
    with open('./ultrachat_200k.jsonl') as f:
        lines = f.readlines()
        for line_txt in lines:
            line = json.loads(line_txt)
            if 'image' in line:
                if 1 != line['conversations'][0]['value'].count('<image>'):
                    continue
            if 'images' in line:
                if len(line['images']) != line['conversations'][0]['value'].count('<image>'):
                    continue
            if line['conversations'][0]['value'].count('<image>') != 0 and 'image' not in line and 'images' not in line:
                continue
            fm.write(json.dumps(line)+'\n')

with open('./ultrachat_200k_fix.jsonl', 'r') as f2:
    lines = f2.readlines()
    print(len(lines))




# from glob import glob
# import json


# def fix_images(str,n):
#     # 初始化计数器
#     count = 0

#     # 初始化新字符串
#     new_str = ""
#     substr = '<image>'

#     # 遍历字符串，替换子串
#     for i in range(len(str)):
#         # 如果找到子串并且计数器小于n
#         if str[i:i+len(substr)] == substr and count < n:
#             # 添加子串到新字符串
#             new_str += substr
#             # 增加计数器
#             count += 1
#             # 跳过子串的剩余部分
#             i += len(substr) - 1
#         else:
#             # 添加字符到新字符串
#             new_str += str[i]
#     return new_str

# def func(lines):
#     for i in range(len(lines)):
#         line = lines[i]
#         a = json.loads(line)
#         count = a['conversations'][0]['value'].count('<image>')
#         lens = 0
#         if 'image' in a:
#             lens += 1
#         if 'images' in a:
#             lens += len(a['images'])
#         if count != lens:
#             lines[i] = fix_images(a['conversations'][0]['value'],len(a['images']))
#     return lines


# for file in glob('./general_mm_exam.jsonl'):
#     with open(file) as f:
#         lines = f.readlines()
#         lines = func(lines)
#     with open(file,'w') as f:
#         for line in lines:
#             f.write(json.dumps(line,ensure_ascii=False)+'\n')

