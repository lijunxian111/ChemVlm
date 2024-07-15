import os
from glob import glob
import random
import PIL
from tqdm import tqdm
import cv2
import numpy as np
import torch
from rxnscribe import RxnScribe

from huggingface_hub import hf_hub_download

REPO_ID = "yujieq/RxnScribe"
FILENAME = "pix2seq_reaction_full.ckpt"
ckpt_path = hf_hub_download(REPO_ID, FILENAME)

device = torch.device('cpu')
model = RxnScribe(ckpt_path, device)


def get_markdown(reaction):
    output = []
    for x in ['reactants', 'conditions', 'products']:
        s = f'The {x} of this reaction were,\n'
        for ent in reaction[x]:
            if 'smiles' in ent:
                s += "\n```smiles\n" + ent['smiles'] + "\n```\n"
            elif 'text' in ent:
                s += ' '.join(ent['text']) + '\n'
            else:
                s += ent['category']
        output.append(s)
    return '\n'.join(output)


def predict(image, molscribe=True, ocr=True):
    predictions = model.predict_image(image, molscribe=molscribe, ocr=ocr)
    # pred_image = model.draw_predictions_combined(predictions, image=image)
    markdown = f'There were {len(predictions)} reactions in this picture.\n'+'\n'.join([f'{i}.\n'+get_markdown(reaction) for i, reaction in enumerate(predictions)])
    return markdown

pngs = glob('/mnt/hwfile/ai4chem/zhangdi/rxn_images/images/*.png')
random.shuffle(pngs)

for file in tqdm(pngs):
    if os.path.exists(file.replace('.png', '.md')):
        continue
    if os.path.exists(file.replace('.png', '.lock')):
        continue
    else:
        os.system(f"touch {file.replace('.png', '.lock')}")
    image = PIL.Image.open(file).convert('RGB')

    md = predict(image)
    with open(file.replace('.png', '.md'), 'w') as f:
        f.write(md)
    os.remove(file.replace('.png', '.lock'))