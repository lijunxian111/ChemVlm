import os

n = 64
for i in range(n):
    os.system('sbatch -p AI4Phys --quotatype=auto --cpus-per-task=8 --gres=gpu:0 /mnt/petrelfs/zhangdi1/lijunxian/datagen/cross_modal/rxn/batch.sh')