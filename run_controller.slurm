#!/bin/bash
#SBATCH --job-name=ctrl_train
#SBATCH --account=henderson
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --time=00:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load anaconda3/2024.2
module load proxy/default
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/trl

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
export PATH=/scratch/gpfs/HENDERSON/zs7353/envs/trl/bin:$PATH

# Set cache dirs to scratch to avoid quota issues
export HF_HOME=/scratch/gpfs/HENDERSON/zs7353/legacy/.cache/huggingface
export TORCH_HOME=/scratch/gpfs/HENDERSON/zs7353/legacy/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure proper GPU visibility for distributed training
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# OpenAI API key for GPT-5 judge
export OPENAI_API_KEY="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"

# Run the training script with DeepSpeed via accelerate
echo "Starting Controller Training with DeepSpeed..."
accelerate launch \
    --config_file accelerate_config.yaml \
    --mixed_precision=no \
    train_controller_standalone.py \
    --response-length 1024 \
    --logging-steps 1 \
    --save-steps 50


