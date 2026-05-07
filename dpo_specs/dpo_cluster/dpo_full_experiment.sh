#!/bin/bash
#SBATCH --job-name=dpo_full
#SBATCH --account=rarora8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/weka/scratch/rarora8/madhu/dpo_experiment/logs/dpo_full_%j.out
#SBATCH --error=/weka/scratch/rarora8/madhu/dpo_experiment/logs/dpo_full_%j.err

module load anaconda3/2024.02-1
conda activate rlvr

mkdir -p /weka/scratch/rarora8/madhu/dpo_experiment/logs
mkdir -p /weka/scratch/rarora8/madhu/dpo_experiment/results

cd /weka/scratch/rarora8/madhu/dpo_experiment

python dpo_full_experiment.py
