#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J Train_and_Evaluate
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Uno GPU exclusivo por favor
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
#BSUB -M 5GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
##BSUB -u s214591@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_train_and_eval.out
#BSUB -e Output_train_and_eval.err

# Activate the Python environment
source /work3/s214591/miniconda3/etc/profile.d/conda.sh
conda activate /work3/s214591/miniconda3/envs/deepLearningProject

# Navigate to the directory containing the train_and_evaluate.py script
cd /work3/s214591/path_to_your_script_directory

# Run the train_and_evaluate.py script
python train_and_evaluate.py

