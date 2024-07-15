#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=AI4Chem
##SBATCH -w SH-IDC1-10-140-24-109

source /mnt/petrelfs/zhangdi1/miniforge3/bin/activate internvl_new
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
export all_proxy=""
cd /mnt/petrelfs/zhangdi1/lijunxian/InternVL/internvl_chat
#python -m internvl.serve.model_worker --host 0.0.0.0 --controller http://0.0.0.0:10050 --port 10077 --worker http://0.0.0.0:10077 --model-path /mnt/hwfile/ai4chem/CKPT/wxz/chemvl_2B_ft_7_3_0_merge --device auto --model-name chemvlm_2B

python -m internvl.serve.model_worker --host 0.0.0.0 --controller http://10.140.24.69:10050 --port 10077 --worker http://10.140.24.69:10077 --model-path /mnt/hwfile/ai4chem/CKPT/wxz/chemvl_2B_ft_7_3_0_merge --device auto --model-name chemvlm_2B