from glob import glob
import os

files = glob('./*.jsonl')

for file in files:
    print(file.split('/')[-1])
    os.system(f'wc -l {file}')