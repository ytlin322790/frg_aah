#!/usr/local_rwth/bin/zsh
for i in $(seq 0 1 10)
do
for ii in $(seq 0 1 5)
do
for site in $(seq 0 400 2000)
do


touch run_individual

echo "#!/usr/local_rwth/bin/bash" >> run_individual
echo "#SBATCH --job-name=aah$i.$ii.$site" >> run_individual
echo "#SBATCH --output=Out/MOD$i.$ii.$site.dat" >> run_individual
echo "#SBATCH --time=2:00:00 " >> run_individual
echo "#SBATCH --mem=2000 " >> run_individual
echo "#SBATCH --nodes=1" >> run_individual
echo "#SBATCH --account=rwth0444" >> run_individual
echo "#SBATCH --partition=c18m" >> run_individual
echo "module load pythoni/3.7" >> run_individual
echo "cd $HOME/SSH" >> run_individual
echo "python3 job.py $i $ii $site" >> run_individual

chmod +x run_individual
sbatch run_individual
rm run_individual

done
done
done
