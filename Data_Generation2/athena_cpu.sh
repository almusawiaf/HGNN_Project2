#!/bin/bash

#SBATCH --output=logs/X_output_%j.txt           # Standard output
#SBATCH --error=logs/X_error_%j.txt             # Standard error

#SBATCH --nodes=1                          # Request one node
#SBATCH --ntasks=1                         # Request one task (process)

#SBATCH --cpus-per-task=16                 # Number of CPU cores per task
#SBATCH --mem=250G                         # Allocate memory (512 GB in this case)

#SBATCH --job-name=data_generation_15000

# Set environment variables
export MIMIC_Path='/home/almusawiaf/MyDocuments/PhD_Projects/Data/MIMIC_resources'
export disease_data_path='../Data'

export NUM_DISEASES=203
export DISEASE_FILE='DMPLB2'
export similarity_type='PC'

export num_Sample=10010
export r_u_sampling='True'
export PSGs_ing='True'
experiment_name="10K_patients_PSGs"


output_dir="/lustre/home/almusawiaf/PhD_Projects/HGNN_Project2/Data_Generation2/experiments"
mkdir -p $output_dir

jupyter nbconvert --to notebook --execute main_cpu.ipynb                  --output $output_dir/main_cpu_${NUM_DISEASES}_${num_Sample}_${DISEASE_FILE}_${experiment_name}.ipynb
