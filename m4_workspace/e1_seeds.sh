#!/bin/bash
#SBATCH --job-name=e1
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=42,43,44
#SBATCH --output=/home/%u/logs/%x-%a-%j.out
module load python
source /scratch/$USER/envs/stocknet/bin/activate
cd ~/project/dl/m4_workspace
python train.py --price-encoder gru --seed $SLURM_ARRAY_TASK_ID --run-name e1_gpu_seed$SLURM_ARRAY_TASK_ID
