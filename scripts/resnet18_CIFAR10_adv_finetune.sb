#!/bin/bash --login
########### SBATCH Lines for Resource Request ##########


#SBATCH --time=1:40:00            # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --exclude=lac-143
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=4           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=2G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name hydra_resnet18_CIFAR10_separate_seed    # you can give your job a name for easier identification (same as -J)
#SBATCH --gres=gpu:v100:1
#SBATCH --output=log/slurm/resnet18_CIFAR10_separate_seed.out     # modify it to the name you want for output
########## Command Lines to Run ##########

module purge
module load GCC/6.4.0-2.28  OpenMPI/2.1.2
module load CUDA/10.0.130 cuDNN/7.5.0.56-CUDA-10.0.130
module load Python/3.8.5

export PATH=$PATH:$HOME/anaconda3/bin
source activate biprune
cd ~/hydra-structured
python3 train.py --arch resnet18 --dataset CIFAR10 --k ${k} --exp-mode finetune --exp-name resnet18_ratio${k}_adv --trainer adv --val-method adv --source-net results/resnet18/resnet18_ratio${k}_adv/prune/latest_exp/checkpoint/model_best.pth.tar --result-dir results
scontrol show job $SLURM_JOB_ID     ### write job information to output file

# Submission command:
# k=0.8
# sbatch --job-name=resnet18_CIFAR10_k${k}_adv_finetune --output=log/slurm/resnet18_CIFAR10_k${k}_adv_finetune.log --export=k=${k} scripts/resnet18_CIFAR10_adv_finetune.sb