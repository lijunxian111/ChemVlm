import json
import random

with open('./text_pure.jsonl') as f:
    lines = f.readlines()
    sampled_line = []
    random.shuffle(lines)
    with open('./text_pure_sample.jsonl', 'w') as f2:
        for i in random.sample(range(0,len(lines)),len(lines)):
            if '填空题' not in lines[i] and '单选题' not in lines[i] and '简答题' not in lines[i] and '解答题' not in lines[i]:
                print(lines[i])
                raise 999