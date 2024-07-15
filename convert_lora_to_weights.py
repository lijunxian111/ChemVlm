from peft import PeftModel
import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from chemexam_repo.ChemLLM_Multimodal_Exam.internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from chemexam_repo.ChemLLM_Multimodal_Exam.internvl.train.dataset import build_transform, dynamic_preprocess
# from internvl.model.internvl_chat import (InternVisionConfig,
#                                           InternVisionModel,
#                                           InternVLChatConfig,
#                                           InternVLChatModel)
# from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, LlamaConfig, LlamaForCausalLM,
                          LlamaTokenizer, Trainer, TrainingArguments,
                          default_data_collator, set_seed)
from accelerate import infer_auto_device_map
import accelerate
import pdb

if __name__ == "__main__":
    model = InternVLChatModel.from_pretrained('/mnt/hwfile/ai4chem/CKPT/chemvl_pt_6_1_0/checkpoint-200/')
    print("Loading lora")
    #model = PeftModel.from_pretrained(model, '/mnt/hwfile/ai4chem/share/multimodal-exam/internvl_chemllm-projector_lr_1e-3')
    # pdb.set_trace()
    # model.language_model = model.language_model.merge_and_unload()
    model.vision_model = model.vision_model.merge_and_unload()
    model.save_pretrained('/mnt/hwfile/ai4chem/CKPT/chemvl_pt_6_1_0/checkpoint-200-merge/')
    # print("Lora model is loaded")