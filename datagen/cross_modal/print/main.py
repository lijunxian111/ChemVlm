from glob import glob
import json
import random

files = glob('/mnt/hwfile/ai4chem/zhangdi/real_mole_images/real/*/*.png',recursive=True)

chineses = [
    "请问这张图片展示的分子名称是什么？",
    "这个分子结构代表哪种化合物？",
    "能否帮我识别图中的这个分子？",
    "这张图像中的化学结构是什么？",
    "这个分子的化学式能告诉我吗？",
    "图中所示的是哪种分子？",
    "您能解释一下图片中分子的组成吗？",
    "这张图片中的是哪一种化学分子？",
    "您能识别出图片中的这个分子吗？",]

english = [
    "What is the name of the molecule shown in this image?",
"What kind of compound does this molecular structure represent?",
"Can you help me identify this molecule in the image?",
"What is the chemical structure in this image?",
"Can you tell me the chemical formula of this molecule?",
"What kind of molecule is shown in the image?",
"Can you explain the composition of the molecule in the image?",
"What kind of chemical molecule is in this image?",
"Can you identify the molecule in the image?",
]

expressions = [
    "我认为这张图片里的分子式，用SMILES表示应为 {SMILES}。",
    "在我看来，图片中的分子结构用SMILES格式写作 {SMILES}。",
    "我的看法是，该图显示的分子式的SMILES表示法是 {SMILES}。",
    "根据我的理解，图中的分子式可用SMILES形式表示为 {SMILES}。",
    "我觉得，这个图片中的化学结构，用SMILES来描述，就是 {SMILES}。",
    "从我个人的角度出发，图片里的分子结构用SMILES可以表示为 {SMILES}。",
    "据我观察，此图中的分子式以SMILES形式表达，应为 {SMILES}。",
    "我的解读是，这幅图中的分子式，按SMILES格式是 {SMILES}。",
    "以我的见解，这张图片展示的分子式的SMILES表达应该是 {SMILES}。",
    "对我来说，该图片中分子的SMILES表示应是 {SMILES}。",
    "我的分析显示，这张图中的分子式以SMILES格式表示为 {SMILES}。",
    "在我看来，这张图片中的分子结构的SMILES表达式是 {SMILES}。",
    "依我分析，图中展示的分子式如果用SMILES写出来应是 {SMILES}。",
    "我猜想这幅图片的分子式，如果用SMILES语法来表达，会是 {SMILES}。",
    "从我的角度判断，图片中所示分子式的SMILES形式是 {SMILES}。",
    "我推测这张图片中的分子式，以SMILES格式表达，可能是 {SMILES}。",
    "依据我的观察，这个图片展示的分子式，按SMILES来说是 {SMILES}。",
    "我估计这幅图的分子结构，用SMILES方式写出来，应当是 {SMILES}。",
    "我的意见是，该图的分子结构以SMILES格式表示则为 {SMILES}。",
    "在我看来，图中的分子式转换成SMILES语言应该是 {SMILES}。",
    "依我之见，这个图片中的分子结构，SMILES格式表示为 {SMILES}。",
    "我的理解是，图中的分子结构，用SMILES代码表示，是 {SMILES}。",
    "从我的视角，该图片中的分子式，如果用SMILES表示，会是 {SMILES}。",
    "从我的分析来看，这张图片中的分子式用SMILES语法表达，为 {SMILES}。",
    "我的判断是，这幅图展示的分子结构，用SMILES来写应是 {SMILES}。",
    "我认为这张图片中的化学结构，转换成SMILES格式，就是 {SMILES}。",
    "我感觉这张图中的分子式，以SMILES的方式表示出来，是 {SMILES}。",
    "我认为这幅图表中的分子结构，用SMILES语法表示，就是 {SMILES}。",
    "从科学的视角来看，这张图片中的分子式用SMILES标记，应是 {SMILES}。",
    "我相信这张图片展示的分子结构，采用SMILES格式来表达，将是 {SMILES}。"
]


english_expressions = [
    "I believe the molecular formula in this image, when represented with SMILES, should be {SMILES}.",
    "From my perspective, the molecular structure in the image is written in SMILES format as {SMILES}.",
    "In my view, the molecular formula displayed in the image with a SMILES representation is {SMILES}.",
    "According to my understanding, the molecular formula in the image can be represented in SMILES as {SMILES}.",
    "I think the chemical structure in this image, when described with SMILES, is {SMILES}.",
    "From my personal perspective, the molecular structure in the image can be represented with SMILES as {SMILES}.",
    "As I observe, the molecular formula in this image, expressed in SMILES format, should be {SMILES}.",
    "My interpretation is that the molecular formula in this image, according to SMILES format, is {SMILES}.",
    "In my opinion, the SMILES expression of the molecular formula shown in this image should be {SMILES}.",
    "For me, the SMILES representation of the molecule in the image should be {SMILES}.",
    "My analysis shows that the molecular formula in this image is represented in SMILES format as {SMILES}.",
    "In my view, the molecular structure in this image is expressed in SMILES notation as {SMILES}.",
    "Based on my analysis, the molecular formula displayed in the image, if written in SMILES, would be {SMILES}.",
    "I guess the molecular formula in this image, if expressed in SMILES syntax, would be {SMILES}.",
    "From my angle, the molecular formula in the image in SMILES format is {SMILES}.",
    "I speculate that the molecular formula in this image, expressed in SMILES format, could be {SMILES}.",
    "Based on my observation, the molecular formula in this image, as per SMILES, is {SMILES}.",
    "I estimate that the molecular structure in this image, written in SMILES, should be {SMILES}.",
    "My opinion is that the molecular structure of the image in SMILES format is {SMILES}.",
    "In my view, the molecular formula in the image translated into SMILES language should be {SMILES}.",
    "From my perspective, the molecular structure in this image, represented in SMILES format, is {SMILES}.",
    "My understanding is that the molecular structure in the image, represented with SMILES code, is {SMILES}.",
    "From my vantage point, the molecular formula in the image, if represented with SMILES, would be {SMILES}.",
    "From my analysis, the molecular formula in this image expressed in SMILES syntax is {SMILES}.",
    "My judgment is that the molecular structure displayed in the image, written in SMILES, should be {SMILES}.",
    "I believe the chemical structure in this image, when converted to SMILES format, is {SMILES}.",
    "I feel that the molecular formula in this image, represented in SMILES fashion, is {SMILES}.",
    "I believe that the molecular structure in this chart, if represented with SMILES syntax, is {SMILES}.",
    "From a scientific viewpoint, the molecular formula in this image, marked with SMILES, should be {SMILES}.",
    "I am confident that the molecular structure shown in this image, expressed in SMILES format, would be {SMILES}."
]


import pandas as pd
labels = {}

for csv in glob('/mnt/hwfile/ai4chem/zhangdi/real_mole_images/real/*.csv'):
    with open(csv) as f:
        for line in f:
            index,path,smiles = line.strip().split(',')
            labels[path] = smiles

def get_label(file):
    index = file.replace('/mnt/hwfile/ai4chem/zhangdi/real_mole_images/','')
    return labels[index]


def chinese_style(label):
    return random.choice(expressions).format(SMILES=label)

def english_style(label):
    return random.choice(english_expressions).format(SMILES=label)

def add_image_token(text):
    a = random.choice('01')
    if a == '0':
        text = '<image>\n' + text
    else:
        text = text + '\n<image>'
    return text

with open('data.jsonl','w') as f:
    print(len(files))
    for i,file in enumerate(files):
        lang = random.choice(['english','chinese'])
        conv = []
        if lang == 'chinese':
            q = {'from':'human','value':add_image_token(random.choice(chineses))}
            a = {'from':'gpt','value':chinese_style(get_label(file))}
        else:
            q = {'from':'human','value':add_image_token(random.choice(english))}
            a = {'from':'gpt','value':english_style(get_label(file))}
        
        conv = [q,a]
        jsons = {'id':i,'image':file,'conversations':conv}
        f.write(json.dumps(jsons,ensure_ascii=False)+'\n')

