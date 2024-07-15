# 读取PIL，包
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os


ls = ['okvqa',  'clevr_math','ai2d', 'aokvqa', 'chart2text', 'chartqa', 'clevr',  'cocoqa', 'datikz', 'diagram_image_to_text', 'docvqa', 'dvqa', 'figureqa', 'finqa', 'geomverse', 'hateful_memes', 'hitab', 'iam', 'iconqa', 'infographic_vqa', 'intergps', 'localized_narratives', 'mapqa', 'mimic_cgd', 'multihiertt', 'nlvr2', 'ocrvqa', 'plotqa', 'raven', 'rendered_text', 'robut_sqa', 'robut_wikisql', 'robut_wtq', 'scienceqa', 'screen2words', 'spot_the_diff', 'st_vqa', 'tabmwp', 'tallyqa', 'tat_qa', 'textcaps', 'textvqa', 'tqa', 'vistext', 'visual7w', 'visualmrc', 'vqarad', 'vqav2', 'vsr', 'websight']

idx = 0
for subset in tqdm(ls):
    print("# 子集", subset)
    data = load_dataset("/mnt/hwfile/ai4chem/share/the_cauldron", subset, split='train')

    print(data)

    # {'images': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=299x227 at 0x7FA37BD179D0>], 'texts': [{'user': 'Question: What do respiration and combustion give out\nChoices:\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Heat\nAnswer with the letter.', 'assistant': 'Answer: B', 'source': 'AI2D'}]}
    dir_path = Path('/mnt/hwfile/ai4chem/share/the_cauldron_images')
    for i in tqdm(range(len(data))):
        # 处理图像
        try:
            row = data[i]
            for ix,img in enumerate(row['images']):
                img_path = dir_path.joinpath(f'{idx}_cauldron_{subset}_{ix}.png')
                if os.path.exists(img_path):
                    continue
            
                img.save(img_path)
                idx += 1

        except Exception as e:
            print(e)