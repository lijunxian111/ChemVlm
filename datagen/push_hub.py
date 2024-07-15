from datasets import load_dataset

dataset = load_dataset('json',data_files='./text_pure.jsonl')

dataset.push_to_hub('AI4Chem/ChemExam_puretext')