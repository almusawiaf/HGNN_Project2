#!/bin/bash

#SBATCH --output=logs/output_%j.txt            # Standard output
#SBATCH --error=logs/error_%j.txt              # Standard error

#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task (process)
#SBATCH --cpus-per-task=40                # Request four CPU cores per task
#SBATCH --qos=short
#SBATCH --mem=2800G


#SBATCH --job-name=SAGE

# Set environment variables
export num_Sample=54545
export NUM_DISEASES=203
export NUM_TOP_DISEASES=10
export DISEASE_FILE='DMPLB2'
export num_Meta_Path=35
export num_epochs=1000
export experiment_name='DMPLB2_1000_epochs2_similarity_added'

# output_dir="/lustre/home/almusawiaf/PhD_Projects/HGNN_Project2/GNN_Models/experiments/${NUM_DISEASES}_Diagnoses/${DISEASE_FILE}/${num_Sample}"
output_dir="experiments/${NUM_DISEASES}_Diagnoses/${DISEASE_FILE}/${num_Sample}"
mkdir -p $output_dir


# Run the second notebook in the background
# jupyter nbconvert --to notebook --execute GCN.ipynb --output=$output_dir/X_GCN.ipynb 
jupyter nbconvert --to notebook --execute SAGE.ipynb --output=$output_dir/X_SAGE_${DISEASE_FILE}_${NUM_TOP_DISEASES}D_${experiment_name}.ipynb 
