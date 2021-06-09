#!/bin/sh
#SBATCH -v
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=vincenzo.schimmenti@universite-paris-saclay.fr
#SBATCH --job-name=isostress
#SBATCH --time=23:59:59
##SBATCH --mem-per-cpu=5G
eval $1 ; eval $2 ; eval $3 ; eval $4 ; eval $5 ; eval $6 ;

echo  "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

python train.py --input_dir=$input_dir --output_dir=$output_dir --dataset_list=$dataset_list --model_filename=vdep_unet_sh.dict --batch_size=5 --epochs=500 --evaluate=$evaluation --regression=$regression --train_split=0.7 --lr=$lr >> log.txt
