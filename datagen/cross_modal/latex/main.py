import json
import random
from datasets import load_dataset
from tqdm import tqdm

dataset0 = load_dataset("OleehyO/latex-formulas",'cleaned_formulas',split='train')


english = [
    "Could you help me convert the content of this image into LaTeX format?",
    "Is it possible for you to transform the information in this image into LaTeX format?",
    "Can you assist me in converting what's in this image to LaTeX?",
    "Would you be able to turn the contents of this image into LaTeX code?",
    "Could you change the content of this image into LaTeX format for me?",
    "I need help with converting this image to LaTeX format. Can you do that?",
    "Can this image's content be rewritten into LaTeX format?",
    "Are you able to help me render the contents of this image in LaTeX?",
    "Please could you convert what’s shown in this image into LaTeX?",
    "Could you handle the conversion of this image’s contents to LaTeX format?",
    "Is converting the content of this image into LaTeX something you can do?",
    "I'd appreciate your assistance in converting this image to LaTeX format.",
    "Can you manage to encode the contents of this image into LaTeX?",
    "Would it be possible to translate the content of this image into LaTeX?",
    "Can you help transform the information shown in this image into LaTeX format?"
]

chineses = [
    "你能帮我把这张图的内容转成LaTeX格式吗？",
    "请问你能把这张图片转换为LaTeX格式吗？",
    "能帮我将这图里的内容做成LaTeX格式吗？",
    "你能将这张图片的内容改写为LaTeX代码吗？",
    "能否请你帮忙把这图片内容转为LaTeX格式？",
    "你能帮忙把这图片里的信息转换成LaTeX格式吗？",
    "这张图片的内容可以转成LaTeX格式吗？",
    "请帮我将这图中的内容制作成LaTeX格式。",
    "能把这张图的内容转化为LaTeX形式吗？",
    "你能处理这张图片内容转成LaTeX格式的任务吗？",
    "你有办法把这张图片的内容变成LaTeX格式吗？",
    "我需要这张图片转成LaTeX格式，可以吗？",
    "请将这图的内容编写成LaTeX格式。",
    "这张图片能转换成LaTeX格式的代码吗？",
    "可以将这张图片的信息改写为LaTeX格式吗？"
]

ans_chinese = [
    "当然可以，用LaTeX表达这张图片的内容会是 {latex}",
    "毫无问题，这张图片里的内容用LaTeX来写应该是 {latex}",
    "可以的，将这张图片的内容用LaTeX格式写出来是 {latex}",
    "当然行，这张图片的内容如果用LaTeX写，会是 {latex}",
    "绝对可以，用LaTeX将这张图片的内容表示出来就是 {latex}",
    "没问题，这张图片的LaTeX表达形式是 {latex}",
    "可以呀，把这张图片的内容转化为LaTeX的话，应该写作 {latex}",
    "确实可以，这张图片内容用LaTeX形式展示是 {latex}",
    "当然能，按LaTeX格式，这张图片的内容可以写为 {latex}",
    "可以的，这张图片内容若用LaTeX编写，其形式是 {latex}",
    "肯定可以，用LaTeX把这张图片的内容表述出来会是 {latex}",
    "可以，这张图片的内容用LaTeX描述就是 {latex}",
    "当然可以实现，这张图片用LaTeX书写的正确形式是 {latex}",
    "行的，以LaTeX的方式将这张图片的内容写下来，将是 {latex}",
    "当然无妨，将这图片的内容用LaTeX形式呈现出来，就是 {latex}"
]

ans_english = [
    "Of course, the content of this image can be expressed in LaTeX as {latex}.",
    "Certainly, this image's content can be written in LaTeX format as {latex}.",
    "Absolutely, the content from this image can be represented in LaTeX by {latex}.",
    "Sure, if you want to write this image's content in LaTeX, it would be {latex}.",
    "Yes, translating this image into LaTeX would result in {latex}.",
    "Indeed, you can render the content of this image in LaTeX as {latex}.",
    "Definitely, this image's content in LaTeX would look like {latex}.",
    "Yes, you can write the content of this image as {latex} in LaTeX.",
    "Of course, the LaTeX version of this image's content would be {latex}.",
    "Certainly, if we transcribe this image to LaTeX, it would appear as {latex}.",
    "Surely, converting this image's content to LaTeX, you would get {latex}.",
    "Absolutely, this image can be translated into LaTeX as {latex}.",
    "Indeed, the LaTeX representation of this image's content is {latex}.",
    "Yes indeed, the LaTeX code for this image's content is {latex}.",
    "Definitely, to express this image's content in LaTeX, use {latex}."
]

def imwrite(path,img):
    try:
        img.save(path)
    except:
        img = img.convert('RGBA')
        img.save(path)

def add_image_token(text):
    a = random.choice('01')
    if a == '0':
        text = '<image>\n' + text
    else:
        text = text + '\n<image>'
    return text

with open('./data.jsonl','w') as fm:
    dataset = dataset0

    # for data in dataset:
    for i in tqdm(random.sample(range(0,len(dataset)),100*1000)):
        data = dataset[i]
        lang = random.choice(['english','chinese'])
        conv = []
        if lang == 'chinese':
            q = {'from':'human','value':add_image_token(random.choice(chineses))}
            a = {'from':'gpt','value':random.choice(ans_chinese).format(latex=data['latex_formula'])}
        else:
            q = {'from':'human','value':add_image_token(random.choice(english))}
            a = {'from':'gpt','value':random.choice(ans_english).format(latex=data['latex_formula'])}
       
        conv = [q, a]
        ids = id(q)
        imwrite('images/'+str(ids)+'.png', data['image'])
        out = {'id':ids, 'images':['images/'+str(ids)+'.png',],'conversations':conv,}
        fm.write(json.dumps(out)+'\n')

    # subsets = ['default','equation','figure']

    # for subset in subsets:
    #     dataset = load_dataset("JosselinSom/Latex-VLM",subset,split='train+validation')
    #     for data in tqdm(dataset):
    #         lang = random.choice(['english','chinese'])
    #         conv = []
    #         if lang == 'chinese':
    #             q = {'from':'gpt','value':add_image_token(random.choice(chineses))}
    #             a = {'from':'human','value':random.choice(ans_chinese).format(latex=data['tex_code'])}
    #         else:
    #             q = {'from':'gpt','value':add_image_token(random.choice(english))}
    #             a = {'from':'human','value':random.choice(ans_english).format(latex=data['tex_code'])}
        
    #         conv = [q, a]
    #         ids = id(q)
    #         imwrite('images/'+str(ids)+'.png', data['output'])
    #         out = {'id':ids, 'images':['images/'+str(ids)+'.png',],'conversations':conv,}
    #         fm.write(json.dumps(out)+'\n')
