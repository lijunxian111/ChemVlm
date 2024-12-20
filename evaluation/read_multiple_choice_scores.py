import json
import numpy as np
import os

def cal_choice_scores(path: str):
    with open(path, 'r') as f:
        json_data = f.readlines()
    
    cnt = 0.
    total_cnt = 0.
    for index, line in enumerate(json_data):
        line = json.loads(line)
        if line['text'] == line['annotation']:
            cnt +=1 
          
        elif isinstance(line['text'], str) and "".join(re.findall("[A-Z]", line['text'])) == line['annotation']:
            cnt += 1
        
        elif isinstance(line['text'], list):
            gpt_chose = line['text'][0]['text']
            if gpt_chose == line['annotation']:
                cnt += 1
            elif "".join(re.findall("[A-Z]", gpt_chose)) == line['annotation']:
                cnt += 1
        total_cnt+=1
    print(cnt)
    print(total_cnt)
    print(cnt/total_cnt)
    return
