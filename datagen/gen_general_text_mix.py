import json
from datasets import load_dataset

# dataset = load_dataset("HuggingFaceH4/ultrachat_200k",split="train_sft")

# with open('./ultrachat_200k.jsonl','w+') as f:
#     for data in dataset:
#         flag = True
#         if '<image>' in data['prompt']:
#             flag = False
#             continue
#         conv = [{'from':'human','value':data['prompt']},]
#         for item in data['messages']:
#             if '<image>' in item:
#                 flag = False
#                 continue
#             if item['role'] == 'user' :
#                 conv.append({'from':'human','value':item['content']})
#             else:
#                 conv.append({'from':'gpt','value':item['content']})
#         if flag:
#             jsons = {
#                 'id' :id(data),
#                 'conversations':conv
#             }
#             f.write(json.dumps(jsons)+'\n')


from datasets import load_dataset

dataset = load_dataset("Open-Orca/SlimOrca")['train']

with open('./slimorca.jsonl','w+') as f:
    for data in dataset:
        flag = True
        conv = []
        for item in data['conversations']:
            if '<image>' in item['value'] or item['value'] == '':
                flag = False
                continue
            if item['from'] != 'system' and item['value'] != '':
                conv.append(item)

        if flag:
            jsons = {
                'id' :id(data),
                'conversations':conv
            }
            f.write(json.dumps(jsons)+'\n')

# dataset = load_dataset("AI4Chem/ChemData700K")['train']

# with open('./ChemData700K.jsonl','w+') as f:
#     for data in dataset:
#         conv = []
#         for item in data['history']:
#             q,a = item
#             conv.append({'from':'human','value':q})
#             conv.append({'from':'gpt','value':a})
#         q = data['instruction'] + '\n' + data['input']
#         a = data['output']
#         conv.append({'from':'human','value':q})
#         conv.append({'from':'gpt','value':a})
#         jsons = {
#             'id' :id(data),
#             'conversations':conv
#         }
#         f.write(json.dumps(jsons)+'\n')