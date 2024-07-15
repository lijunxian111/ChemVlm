#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --partition=AI4Chem
##SBATCH -w SH-IDC1-10-140-24-63

export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH

export CUDA_PATH=/mnt/petrelfs/share/cuda-11.8/
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8/
export CUDA_ROOT=/mnt/petrelfs/share/cuda-11.8/s

source /mnt/petrelfs/zhangdi1/miniforge3/bin/activate internvl_new
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
export all_proxy=""
cd /mnt/petrelfs/zhangdi1/lijunxian/InternVL/internvl_chat_llava
python -m llava.serve.controller --host 0.0.0.0 --port 10050