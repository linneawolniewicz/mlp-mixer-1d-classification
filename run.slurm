#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=koa # koa, sadow, gpu, shared, kill-shared, exclusive-long
#SBATCH --account=koa

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=156gb ## max amount of memory per node you require
#SBATCH --time=14-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow, 30day max on koa

#SBATCH --job-name=optuna_mlp_mixer
#SBATCH --output=job.out
#SBATCH --output=logs/slurm_output/job-%A.out
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
source activate pytorch

# use this command to run a python script
# python fno_1d_classification.py 

# use this command to run an ipynb and save outputs in the notebook
# jupyter nbconvert --execute --clear-output fno_1d_classification.ipynb 

# Another command to create a .py script, then run that from a ipynb:
# jupyter nbconvert fno_1d_classification.ipynb --to python
python hyperparameter_tuning.py