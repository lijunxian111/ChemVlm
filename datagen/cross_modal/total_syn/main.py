from glob import glob
import json
import random

files = glob('/mnt/hwfile/ai4chem/zhangdi/total_syn/gifs/*')

chineses = [
    "请问这张图片中合成的是什么？",
    "这张图中的合成目标是什么，涉及哪些药剂？",
    "我很好奇，这张图描述的合成了什么化合物，使用了哪些化学品？",
    "这张图片中正在合成什么物质？用到了哪些药剂？",
    "这张图中正在生产什么？",
    "您能识别出这张图片中合成的是什么吗？",
    "这张图片展示的合成了哪种化学物质，应用了哪些药剂？",
    "这张图片中的合成目标是什么，涉及哪些药剂？",
    "这幅图似乎说明了一个合成过程；产品是什么，？",
    "从这张图片中，您能告诉我合成的是什么吗？",
    "这张图片描绘的合成产品是什么",
    "您能指明这幅图中合成的是什么吗？",
    "这张图片展示的合成过程中涉及哪些药剂？",
    "在这个合成中，目标分子是什么",
    "关于这张图片，正在合成什么物质？"
]

english = [
    "Could you tell me what is being synthesized in this image and which reagents are used?",
    "What is the synthesis target in this diagram, and what reagents are involved?",
    "I'm curious, what compound is this image depicting the synthesis of, and which chemicals are being used?",
    "In this image, what substance is being synthesized and what are the reagents?",
    "What is being produced in this image and what are the required reagents?",
    "Can you identify what is synthesized in this picture and the reagents used?",
    "What chemical is this image showing the synthesis of, and which reagents are applied?",
    "What is the synthesis goal in this image, and what reagents does it involve?",
    "This diagram appears to illustrate a synthesis process; what is the product, and which reagents are utilized?",
    "From this image, can you tell what is being synthesized and the reagents used?",
    "What does this image depict as the synthesis product, and what are the contributing reagents?",
    "Could you specify what is being synthesized in this diagram and the reagents used?",
    "What synthesis process is shown in this image and which reagents are involved?",
    "In the depicted synthesis, what is the target molecule and which reagents are being used?",
    "Regarding this image, what substance is under synthesis and which reagents are employed?"
]

expressions = [
    "我认为这张图片展示的反应路线图是用于合成 {target}，涉及的合成路径为 {path}，使用的药剂包括 {reagent}。",
    "在我看来，图片中的这一反应路线图描述的是合成 {target} 的过程，其经过的路径为 {path}，并使用了 {reagent}。",
    "我的理解是，这个图片中反映的合成路线图旨在制备 {target}，具体路径是 {path}，使用了以下药剂：{reagent}。",
    "根据我的观察，图中所示的合成路线图是为了制造 {target}，通过的合成途径是 {path}，涉及的药剂为 {reagent}。",
    "我觉得这幅图中的反应路线图表明了在合成 {target} 过程中采用的路线是 {path}，所用药剂有 {reagent}。",
    "从这张图片分析，所示的合成路线图似乎指向 {target} 的合成，其路径为 {path}，使用的药剂为 {reagent}。",
    "这个图片表达的合成路线图，从我的角度看，是关于 {target} 的制备，涉及路径 {path} 和药剂 {reagent}。",
    "这幅图展示的反应路线图，据我理解，目的是合成 {target}，涵盖的路径为 {path}，使用的药剂是 {reagent}。",
    "依我之见，此图描绘的合成路线图是为了制备 {target}，包含的合成路径是 {path}，所涉及的药剂有 {reagent}。",
    "这张图片中的反应路线图，以我的理解，是指向 {target} 的合成，包括路径 {path}，和药剂 {reagent}。",
    "我推断这张图中的反应路线图描述的是合成 {target}，具体经过 {path}，使用了 {reagent}。",
    "看这张图，我认为它所示的合成路线图是针对 {target} 的，经过的路径为 {path}，用到的药剂是 {reagent}。",
    "这幅反应路线图，从我的观点出发，显然是为了合成 {target}，经由路径 {path}，涉及药剂 {reagent}。",
    "我相信这张图片所示的合成路线图意在制备 {target}，所采用的路径是 {path}，并用了 {reagent}。",
    "根据这张图片，我理解的合成路线图是向着 {target} 进行，涉及的路径是 {path}，使用的药剂为 {reagent}。",
    "这张图片中所显示的反应路线图，我认为是为合成 {target} 而设计的，包含路径 {path} 和药剂 {reagent}。",
    "在这个图像中，我看到的合成路线图指向 {target} 的制备，涵盖路径 {path} 和药剂 {reagent}。",
    "我的观点是，此图片展示的合成路线图是关于 {target} 的制作，经过 {path}，使用了 {reagent}。",
    "这幅图表中反应路线图的目的，据我看来，是合成 {target}，通过路径 {path}，并利用药剂 {reagent}。",
    "从这张图我理解到的合成路线图是为了达到 {target} 的制备，经由路径 {path}，运用了药剂 {reagent}。",
    "这幅图片描绘的反应路线图显示了合成 {target} 的过程，包括路径 {path} 和所需药剂 {reagent}。",
    "我看到的这张图片中的合成路线图是为了合成 {target}，通过的路径是 {path}，并且使用了药剂 {reagent}。",
    "这个图片所展示的合成路线图，我解读为是为制备 {target}，所经路径 {path}，和所用药剂 {reagent}。",
    "我从这幅图片看到的合成路线图意在说明合成 {target} 的步骤，其中路径是 {path}，使用的药剂包括 {reagent}。",
    "在此图片中，我认为所示的合成路线图是关于 {target} 的制备，涉及路径 {path} 以及使用的药剂 {reagent}。",
    "这张图中的合成路线图，按我的理解，是指合成 {target} 的过程，采用的路径是 {path}，所用药剂是 {reagent}。",
    "从我的分析来看，这张图片中的反应路线图目标是合成 {target}，通过路径 {path}，并应用药剂 {reagent}。",
    "我认为这幅图展示的合成路线图是为了合成 {target}，通过的合成路线是 {path}，使用的药剂是 {reagent}。",
    "从科学的视角来看，这张图片中的合成路线图目的是合成 {target}，路径为 {path}，用到的药剂为 {reagent}。",
    "我相信这张图片中显示的合成路线图是为了制造 {target}，采用的路径是 {path}，涉及的药剂包括 {reagent}。"
]



english_expressions = [
    "I believe this image's reaction pathway diagram is for synthesizing {target}, involving the synthesis path {path}, using reagents such as {reagent}.",
    "From my perspective, the reaction pathway depicted in this image describes the process of synthesizing {target}, through the pathway {path}, using {reagent}.",
    "My understanding is that this image reflects a synthesis route map aimed at preparing {target}, with the specific path being {path} and the reagents used include {reagent}.",
    "According to my observations, the synthesis route map in the image is for making {target}, through the synthesis route {path}, involving reagents {reagent}.",
    "I think this image's reaction pathway diagram indicates the route for synthesizing {target} involves {path} and uses reagents like {reagent}.",
    "Analyzing this image, the depicted synthesis route map seems to target the synthesis of {target}, with the path being {path} and using reagents {reagent}.",
    "This image conveys a synthesis route map, from my view, about the preparation of {target}, involving path {path} and reagents {reagent}.",
    "This diagram, as I understand it, aims to synthesize {target}, covering the path {path} and using reagents such as {reagent}.",
    "In my opinion, this diagram portrays the synthesis route for preparing {target}, including the path {path} and involving reagents {reagent}.",
    "The reaction pathway in this image, to my understanding, points to the synthesis of {target}, including the path {path} and reagents {reagent}.",
    "I infer that the reaction pathway diagram in this image describes the synthesis of {target}, specifically through {path}, using {reagent}.",
    "Looking at this image, I believe the synthesis route map is aimed at {target}, via the path {path}, using reagents {reagent}.",
    "From my point of view, this reaction pathway diagram is clearly for synthesizing {target}, via the route {path}, involving reagent {reagent}.",
    "Based on this image, my understanding of the synthesis route map is geared towards making {target}, involving path {path} and using reagents {reagent}.",
    "In this image, I see a synthesis route map that leads to the preparation of {target}, covering the path {path} and reagents {reagent}."
]


import pandas as pd
labels = {}

def get_label(file):

    txt = open(file.replace('gifs','htmls').replace('.gif','.md')).read()
    target,rxns,agents,_ = txt.split('\n')
    target = target[target.find(':')+1:]
    rxns = rxns[rxns.find(':')+1:].replace('**','')
    agents = agents[agents.find(':')+1:].replace('**','').replace(' • ',', ')
    return {'target':target,'path':rxns,'reagent':agents}


def chinese_style(label):
    return random.choice(expressions).format(**label).replace('  ',' ')

def english_style(label):
    return random.choice(english_expressions).format(**label).replace('  ',' ')

def add_image_token(text):
    a = random.choice('01')
    if a == '0':
        text = '<image>\n' + text
    else:
        text = text + '\n<image>'
    return text

with open('data.jsonl','w') as f:
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

