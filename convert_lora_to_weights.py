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
    model = InternVLChatModel.from_pretrained('YOUR MODEL PATH')
    print("Loading lora")
    #model = PeftModel.from_pretrained(model, 'PROJECTOR')
    # pdb.set_trace()
    # model.language_model = model.language_model.merge_and_unload()
    model.vision_model = model.vision_model.merge_and_unload()
    model.save_pretrained('YOUR MERGED MODEL PATH')
    # print("Lora model is loaded")
