eval $1 ; eval $2 ; eval $3 ;

source_filename=/mnt/beegfs/home/schimmenti/corrQuake/viscoelastic/$source

echo "Working on: ${source_filename}$"

n=1
while read line; do
# reading each line
echo "Line No. $n : $line"
sbatch launch_iso.sh source=$line folder=$folder outfolder=$outfolder 
n=$((n+1))
done < $source_filename
