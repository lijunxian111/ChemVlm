from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T

path = "your path" #"OpenGVLab/InternVL-Chat-V1-5" if not special path
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True).eval()
# Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

chemllm = AutoModel.from_pretrained('AI4Chem/ChemLLM-20B-Chat-SFT',torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()

# print(model)
# print(chemllm)

vl_lm = model.language_model
emb_vl = vl_lm.model.tok_embeddings
emb_chem = chemllm.model.tok_embeddings

print(emb_chem.weight.shape)
print(emb_vl.weight.shape)

emb_vl_data = emb_vl.weight.data
emb_chem_data = emb_chem.weight.data

model.language_model.model.tok_embeddings.weight.data[:emb_chem.weight.shape[0],:] = chemllm.model.tok_embeddings.weight.data
model.language_model.model.tok_embeddings.weight.data[:emb_chem.weight.shape[0],:].copy_(chemllm.model.tok_embeddings.weight.data)
model.language_model.output.weight.data[:chemllm.output.weight.shape[0],:] = chemllm.output.weight.data
model.language_model.output.weight.data[:chemllm.output.weight.shape[0],:].copy_(chemllm.output.weight.data)

chemllm.model.tok_embeddings = model.language_model.model.tok_embeddings
chemllm.output = model.language_model.output

print(chemllm.model.tok_embeddings is vl_lm.model.tok_embeddings)

model.language_model = chemllm

print(model.language_model is chemllm)

print(chemllm.model.tok_embeddings is vl_lm.model.tok_embeddings)

model.save_pretrained('./pretrained/chemvlm_26b_not_trained',from_pt=True)
tokenizer.save_pretrained('./pretrained/chemvlm_26b_not_trained',from_pt=True)
