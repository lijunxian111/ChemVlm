#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --partition=AI4Chem
##SBATCH --partition=AI4Phys
##SBATCH -w SH-IDC1-10-140-24-63

export LD_LIBRARY_PATH=cuda-11.8/lib64:$LD_LIBRARY_PATH

export CUDA_PATH=cuda-11.8/
export CUDA_HOME=cuda-11.8/
export CUDA_ROOT=cuda-11.8/s


unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
export all_proxy=""
cd /mnt/petrelfs/zhangdi1/lijunxian/InternVL/internvl_chat_llava
#python -m llava.serve.gradio_web_server --controller http://0.0.0.0:10050 --model-list-mode reload --port 10058

python -m llava.serve.gradio_web_server --controller http://10.140.24.69:10050 --model-list-mode reload --port 10058
