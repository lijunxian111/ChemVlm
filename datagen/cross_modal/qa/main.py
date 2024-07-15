import json
import random
import cv2
from datasets import load_dataset
from tqdm import tqdm

dataset0 = load_dataset("BAAI/CMMU",split='val')

# dataset1 = load_dataset("m-a-p/CMMMU")

dataset2 = load_dataset("derek-thomas/ScienceQA",split='train+validation+test')

dataset2.filter(lambda x: x['image'] != None)

import re

def replace_img_tags(text):
    return re.sub(r'<img="[^"]*">', '<image>', text)

def format_options(options):
    if options is None or len(options) == 0:
        return ''
    ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ret = []
    for i in range(len(options)):
        ret.append(f"{ABC[i]}. {options[i]}")
    return " ".join(ret)

def imwrite(path,img):
    try:
        img.save(path)
    except:
        img = img.convert('RGBA')
        img.save(path)

with open('data.jsonl', 'w') as fm:
    dataset = dataset0
    for data in tqdm(dataset):
        if 'question_info' in data:
            q = data['question_info']
            if '<image>' not in q:
                if random.random() < 0.5:
                    q = q+'\n<image>'
                else:
                    q = '<image>\n'+q
        o = format_options(data['options'])
        a = ",".join(data['answer'])
        e = data['solution_info']
        if a is None:
            a = ''
        if e is None:
            e = ''

        query = {'from':'human','value':q+'\n'+o}
        anser = {'from':'gpt','value':'答案是'+a+'\n'+e}
        conv = [query, anser]
        ids = hash(q)
        imwrite('images/'+str(ids)+'.png', data['image'])
        out = {'id':ids, 'images':['images/'+str(ids)+'.png',],'conversations':conv,}
        fm.write(json.dumps(out)+'\n')

    for subset in ['art_and_design','business','health_and_medicine','humanities_and_social_sciences','science','technology_and_engineering']:
        dataset = load_dataset("m-a-p/CMMMU", subset,split='dev+val+test')
        for data in tqdm(dataset):
            q = data['question']
            q = replace_img_tags(q)
            o = format_options([data['option1'],data['option2'],data['option3'],data['option4']])
            a = ",".join(data['answer'])
            e = data['analysis']
            if e is None:
                e = ''
            e += f"\n这一题的图片类型为{data['img_type']}，考察学科为{data['subcategory']}，难度{data['difficulty_level']}，主要涉及知识点为{data['subfield']}"

            if a is None:
                a = ''
            if e is None:
                e = ''

            query = {'from':'human','value':q+'\n'+o}
            anser = {'from':'gpt','value':'答案是'+a+'\n'+e}
            conv = [query, anser]
            ids = hash(q)
            img_paths = []
            for img in [data['image_1'],data['image_2'],data['image_3'],data['image_4'],data['image_5'],]:
                if img is None:
                    break
                path = 'images/'+str(ids)+str(id(img))+'.png'
                img_paths.append(path)
                imwrite(path, img)
            out = {'id':ids, 'images':img_paths,'conversations':conv,}
            fm.write(json.dumps(out)+'\n')

    dataset = dataset2
    for data in tqdm(dataset):
        if data['image'] is None:
            continue
        q = data['question']
        if '<image>' not in q:
            if random.random() < 0.5:
                q = q+'\n<image>'
            else:
                q = '<image>\n'+q
        o = format_options(data['choices'])
        a = data['answer']
        e = data['hint']

        if a is None:
            a = ''
        if e is None:
            e = ''

        if e is None:
            ans = 'The answer is '+a
        else:
            ans = 'The answer is '+str(a)+'\nExplanation:'+str(e)


        query = {'from':'human','value':q+'\n'+o}
        anser = {'from':'gpt','value':ans}
        conv = [query, anser]
        ids = hash(q)
        imwrite('images/'+str(ids)+'.png', data['image'])
        out = {'id':ids, 'images':['images/'+str(ids)+'.png',],'conversations':conv,}
        fm.write(json.dumps(out)+'\n')