# -*- coding: GBK -*-
import openai

from openai import OpenAI
import os
import base64
import json
 
client = OpenAI(
    api_key="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"
)
 
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
"""
for filename in os.listdir(fig_path):
    if filename.endswith('.png'):
       image_path=os.path.join(fig_path, filename)
       print(image_path)
       base64_image = encode_image(image_path)
       messages=[
        {
            "role": "user", 
             "content": [
                {"type":"text", "text":"What's in this image?"},
                {
                   "type":"image_url",
                   "image_url":{
                      "url":f"data:image/png;base64,{base64_image}"
                      }
                }
            ]
        }
        ]
       completion = client.chat.completions.create(
          model="gpt-4o",
          messages=messages
        )
       chat_response = completion
       answer = chat_response.choices[0].message.content
       print(f'ChatGPT: {answer}')
"""

#prompt = f'你是一位化学教师。现在有一个对题目的解答：```{}```。请根据标准答案```{}```判断这个解答是否正确。你应该回复“正确”或“不正确”。'
#fig_path='Processed'
def test_chemvl_perform(answer_path):
    """
    test our model's performance by gpt-4o
    """
    with open(answer_path, 'r') as f:
        data_to_test = f.readlines()

    total_q_num = len(data_to_test)
    cnt_right_num = 0
    for line in data_to_test:
        line = json.loads(line)
        res = line['text']
        std_ans = line['annotation']
        human_prompt = '你是一位化学教师。现在有一个对题目的解答：```'+res+'```。请根据标准答案```'+std_ans+'```判断这个解答的得分。如果完全正确，请回答“1分”；如果完全错误，请回答“0分”；如果部分正确，请按照正确的比例给出0-1之间的分数。'
        messages=[
        {
            "role": "user", 
             "content": [
                {"type":"text", "text":human_prompt},
            ]
        }
        ]
        completion = client.chat.completions.create(
          model="gpt-4o",
          messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content
        print(f'ChatGPT: {answer}')
        if "正确" in answer and "不正确" not in answer:
            cnt_right_num += 1
        
    print(cnt_right_num/total_q_num)

if __name__ == "__main__":
    test_chemvl_perform("/mnt/petrelfs/zhangdi1/lijunxian/chemexam_repo/ChemLLM_Multimodal_Exam/results/gaokao_chemvl_ft_6_4_0-merge__all_.jsonl")