from datasets import load_dataset

cauldron = load_dataset('/mnt/hwfile/ai4chem/share/the_cauldron','ai2d')

print(cauldron['train'][0]['images'])