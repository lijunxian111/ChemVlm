import json
import random
import sys

# with open('./text_pure.jsonl') as f:
#     lines = f.readlines()
#     sampled_line = []
#     random.shuffle(lines)
#     with open('./text_pure_sample.jsonl', 'w') as f2:
#         for i in random.sample(range(0,len(lines)),100):
#             f2.write(lines[i])

# with open('./text_pure_sample.jsonl', 'r') as f2:
#     lines = f2.readlines()
#     print(len(lines))

# with open('./general_text_pure.jsonl') as f:
#     lines = f.readlines()
#     sampled_line = []
#     random.shuffle(lines)
#     with open('./general_text_pure_sample.jsonl', 'w') as f2:
#         for i in random.sample(range(0,len(lines)),min(165000//4,len(lines))):
#             f2.write(lines[i])

# with open('./general_text_pure_sample.jsonl', 'r') as f2:
#     lines = f2.readlines()
#     print(len(lines))
jsonl = sys.argv[1]
target_len = int(sys.argv[2])
with open(jsonl) as f:
    lines = f.readlines()
    sampled_line = []
    random.shuffle(lines)
    with open(f'{jsonl}.sample.jsonl', 'w') as f2:
        for i in random.sample(range(0,len(lines)),min(len(lines),target_len)):
            f2.write(lines[i])

with open(f'{jsonl}.sample.jsonl', 'r') as f2:
    lines = f2.readlines()
    print(len(lines))