#!/bin/bash
#PBS -l walltime=48:0:0
#PBS -l select=1:ncpus=1:mem=32gb
 
module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/example-env/bin/activate

cd $PBS_O_WORKDIR
python3 -m pip install casadi
python3 /rds/general/user/mfb22/home/CS3/CS3_Training.p