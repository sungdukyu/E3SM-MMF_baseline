#!/bin/bash

# Specify the folder path
folder="/pscratch/sd/h/heroplr/step2_retrain/backup_phase-4_retrained_models"
#folder="/pscratch/sd/s/sungduk/for_neurips/baseline_v2"
# Loop through each file in the folder
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
	#fprocess="${file:47:32}"
	fprocess="${file:69:33}"
	ftag="${file:69:24}"
	#ftag=$(echo $fprocess | sed 's/\.best.h5$//')
        echo $fprocess
	echo $ftag
	sed -e "s/fffffff/${fprocess}/g" Process_R2_Analysis_Step1.py >Process_R2_Analysis_Step1_$ftag.py
        #ncremap -m map_ne4pg2_to_CERES1x1_aave.20230516.nc $file "/pscratch/sd/h/heroplr/R2_analysis_all/"$fsave         
        # Add your desired operations on each file here
        # For example, you can perform actions like copying, moving, or processing the file contents
	chmod 700 Process_R2_Analysis_Step1_$ftag.py
rm submit_Process_R2_$ftag.sh
cat <<EOF >> submit_Process_R2_$ftag.sh 
#!/bin/bash
#SBATCH --job-name="${ftag}_lev"
#SBATCH --output="logs/${ftag}.%j.out"
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --export=ALL
#SBATCH --account=m3312
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liranp@uci.edu
#SBATCH -t 02:00:00

module load python
conda activate Sungduk 
./Process_R2_Analysis_Step1_$ftag.py

EOF

chmod 700 submit_Process_R2_$ftag.sh
sbatch submit_Process_R2_$ftag.sh


    fi
done


#sbatch submit_tend_${varin}_${ilev}.sh

