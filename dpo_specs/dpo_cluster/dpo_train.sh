#!/bin/bash
#SBATCH --job-name=dpo_llama3
#SBATCH --account=rarora8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/rarora8/madhu/dpo_experiment/logs/dpo_%j.out
#SBATCH --error=/scratch/rarora8/madhu/dpo_experiment/logs/dpo_%j.err

module load anaconda3/2024.02-1
conda activate rlvr

cd /scratch/rarora8/madhu/dpo_experiment
python dpo_train.py