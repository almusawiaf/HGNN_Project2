#!/bin/bash

#SBATCH --output=logs/output_%j.txt            # Standard output
#SBATCH --error=logs/error_%j.txt              # Standard error

#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task (process)
#SBATCH --cpus-per-task=40                # Request four CPU cores per task
#SBATCH --qos=short
#SBATCH --mem=2800G

#SBATCH --job-name=data_generation_15000

# Set environment variables
export MIMIC_Path='/home/almusawiaf/MyDocuments/PhD_Projects/MIMIC_resources'
export disease_data_path='../Data'

export NUM_DISEASES=203
export DISEASE_FILE='DMPLB2'
export similarity_type='PC'

export num_Sample=60001
export r_u_sampling='False'
export PSGs_ing='False'
experiment_name="ALL_MIMIC_XY_meta_path_enhanced"


# Create a new directory with the job name and timestamp
output_dir="/home/almusawiaf/MyDocuments/PhD_Projects/HGNN_Project2/Data_Generation2/experiments"

# output_dir="output_${SLURM_JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $output_dir

jupyter nbconvert --to notebook --execute main.ipynb  --output $output_dir/main_${NUM_DISEASES}_${num_Sample}_${DISEASE_FILE}_${experiment_name}.ipynb
