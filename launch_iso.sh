#!/bin/sh
#SBATCH -v
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=vincenzo.schimmenti@universite-paris-saclay.fr
#SBATCH --job-name=isostress


eval $1 ; eval $2 ; eval $3 ;

filename=/mnt/beegfs/home/schimmenti/corrQuake/viscoelastic/$source
program_src=/mnt/beegfs/home/schimmenti/corrQuake/viscoelastic/visco.out
#srun "visco.out" "k0=0.06" "k1=0.0" "k2=1.0" "Lx=512" "Ly=512" "source_file=${filename}"

n=1
while read line; do
# reading each line
echo "Index :  $n : Name : $line"
isosource=/mnt/beegfs/home/schimmenti/corrQuake/viscoelastic/$folder/events0_$line.dat
isoout=/mnt/beegfs/home/schimmenti/corrQuake/viscoelastic/vdep_data/distr0_$line.dat
echo $isosource
srun "visco.out" "k0=0.06" "k1=0.0" "k2=1.0" "Lx=512" "Ly=512" "iso_stress_file=${isosource}" "iso_output_file=${isoout}" "iso_stress_count=3000"
sleep 1
n=$((n+1))
done < $filename
