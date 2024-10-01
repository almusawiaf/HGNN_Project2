#!/bin/bash

#SBATCH --output=%j_output.txt            # Standard output
#SBATCH --error=%j_error.txt              # Standard error

#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task (process)
#SBATCH --cpus-per-task=16                # Request four CPU cores per task
#SBATCH --qos=short
#SBATCH --mem=2800G

#SBATCH --job-name=SAGE_250

# jupyter nbconvert --to notebook --execute SAGE_SNF_patients_only.ipynb --output=xSAGE_SNF_patients_only1.ipynb &

# Run the second notebook in the background
# jupyter nbconvert --to notebook --execute SAGE.ipynb --output=X_5000_100_SAGE_ADAM.ipynb 

# jupyter nbconvert --to notebook --execute complete.ipynb --output=X_complete.ipynb 

python complete.py
# wait


