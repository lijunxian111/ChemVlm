#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=AI4Chem
##SBATCH -w SH-IDC1-10-140-24-109


unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
export all_proxy=""
cd InternVL/internvl_chat
python -m internvl.serve.model_worker --host 0.0.0.0 --controller http://10.140.24.69:10050 --port 10063 --worker http://10.140.24.69:10063 --model-path chemvl_ft_6_19_0_merged --device auto --model-name chemvlm
