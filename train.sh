#!/bin/sh
#SBATCH -v
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:3
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=vincenzo.schimmenti@universite-paris-saclay.fr
#SBATCH --job-name=isostress
#SBATCH --time=23:59:59
##SBATCH --mem-per-cpu=100G
eval $1 ; eval $2 ; eval $3 ; eval $4 ; eval $5 ;

echo  "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"


python train.py --batches_folder=$batches_folder --min_batch_idx=$min_batch_idx --max_batch_idx=$max_batch_idx --model_state_file=$model_state_file --evaluate=$evaluate
