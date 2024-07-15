import os
table = os.popen('squeue -u zhangdi1').read()
for line in str(table).split('\n'):
    print(line)
    if 'batch.sh' in line or 'batch.sh' in line:
        id = line.split('AI4Phys')[0]
        os.system(f'scancel {id}')

os.system('squeue -u zhangdi1')
os.system('rm *.out')