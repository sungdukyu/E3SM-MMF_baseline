#!/bin/bash

# Specify the folder path
folder="/pscratch/sd/h/heroplr/R2_analysis_all"
# Loop through each file in the folder
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
	fsave="${file:38:28}_aave.nc"
        ncremap -m map_ne4pg2_to_CERES1x1_aave.20230516.nc $file "/pscratch/sd/h/heroplr/R2_analysis_all/"$fsave         
        # Add your desired operations on each file here
        # For example, you can perform actions like copying, moving, or processing the file contents
        
    fi
done

