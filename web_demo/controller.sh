#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4


export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH

export CUDA_PATH=cuda-11.8/
export CUDA_HOME=cuda-11.8/
export CUDA_ROOT=cuda-11.8/s

unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
export all_proxy=""
cd InternVL/internvl_chat_llava
python -m llava.serve.controller --host 0.0.0.0 --port 10050
