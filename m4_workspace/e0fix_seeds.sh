#!/bin/bash
#SBATCH --job-name=e0fix
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
python train.py --seed $SLURM_ARRAY_TASK_ID --run-name e0fix_gpu_seed$SLURM_ARRAY_TASK_ID
