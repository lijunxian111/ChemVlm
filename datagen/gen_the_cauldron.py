from datasets import load_dataset

cauldron = load_dataset('./the_cauldron','ai2d')

print(cauldron['train'][0]['images'])
