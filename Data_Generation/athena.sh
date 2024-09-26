#!/bin/bash

#SBATCH --output=X_output_%j.txt           # Standard output
#SBATCH --error=X_error_%j.txt             # Standard error

#SBATCH --nodes=1                          # Request one node
#SBATCH --ntasks=1                         # Request one task (process)

#SBATCH --cpus-per-task=16                 # Number of CPU cores per task
#SBATCH --mem=250G                         # Allocate memory (512 GB in this case)


#SBATCH --job-name=data_generation_All_MIMIC         # Job name

# Set environment variables
export NUM_DISEASES=203
export DISEASE_FILE='DMPLB'
export similarity_type='PC'

export num_Sample=45000
export r_u_sampling='False'

experiment_name="All_MIMIC"

# Create a new directory with the job name and timestamp
output_dir="/lustre/home/almusawiaf/PhD_Projects/HGNN_Project2/Data_Generation/experiments"

# output_dir="output_${SLURM_JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $output_dir

# Pipeline of actions
jupyter nbconvert --to notebook --execute main.ipynb                  --output $output_dir/main_${NUM_DISEASES}_${num_Sample}_${DISEASE_FILE}_${experiment_name}.ipynb
# jupyter nbconvert --to notebook --execute b_data_preparation.ipynb      --output $output_dir/b_data_preparation.ipynb
# # jupyter nbconvert --to notebook --execute c_StructureSimilarity.ipynb --output $output_dir/c_StructureSimilarity.ipynb
# # jupyter nbconvert --to notebook --execute d_SNF.ipynb                 --output $output_dir/d_SNF.ipynb
# # jupyter nbconvert --to notebook --execute e_convert_SNF_to_edge.ipynb --output $output_dir/e_convert_SNF_to_edge.ipynb
# jupyter nbconvert --to notebook --execute f_Y_superclass.ipynb          --output $output_dir/f_Y_superclass.ipynb
