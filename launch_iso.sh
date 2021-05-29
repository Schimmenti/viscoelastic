#!/bin/sh
#SBATCH -v
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:0
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=vincenzo.schimmenti@universite-paris-saclay.fr
#SBATCH --job-name=isostress
#SBATCH --time=23:59:59
#SBTAHC --mem-per-cpu=10G
eval $1 ; eval $2 ; eval $3 ; eval $4 ;

source_folder=/mnt/beegfs/home/schimmenti/corrQuake/viscoelastic/$folder
output_folder=/mnt/beegfs/home/schimmenti/corrQuake/viscoelastic/$outfolder
file_source=/mnt/beegfs/home/schimmenti/corrQuake/viscoelastic/$source
program_src=/mnt/beegfs/home/schimmenti/corrQuake/viscoelastic/visco.out

$program_src k0=0.06 k1=0.0 k2=1.0 Lx=512 Ly=512 source_file=$file_source iso_source_folder=$source_folder iso_output_folder=$output_folder iso_stress_count=3000 iso_resume_from=$resume
