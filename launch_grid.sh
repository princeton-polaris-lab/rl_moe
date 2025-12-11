#!/bin/bash

# Root paths (edit these if you move the project or change model/data locations)
BASE_DIR=/scratch/gpfs/HENDERSON/zs7353/rl_moe
ENV_PATH=/scratch/gpfs/HENDERSON/zs7353/envs/trl
MODEL_PATH=/scratch/gpfs/KOROLOVA/gpt-oss-20b
# Data paths - update these based on DATASET_TYPE
DATASET_TYPE=nemotron  # 'mmlu' for MMLU multiple-choice, 'nemotron' for open-ended
if [ "$DATASET_TYPE" = "mmlu" ]; then
    DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/mmlu
else
    DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/Nemotron-Post-Training-Dataset-v2/data
fi
OUTPUT_ROOT=${BASE_DIR}/controller_rl_grid
LOG_DIR=${BASE_DIR}/logs

# Training hyper-parameters that stay constant across the sweep
NUM_UPDATE_EPOCHS=1       # PPO epochs per batch (1 = fully on-policy)
NUM_TRAIN_EPOCHS=1        # Total training epochs
# PROMPTS_PER_CATEGORY=1000 # for ppo
PROMPTS_PER_CATEGORY=500 # for grpo
MAX_PROMPT_LENGTH=1024
RESPONSE_LENGTH=1024
SEED=42

# Perplexity reward settings
PPL_REWARD_SCALE=1.0      # Perplexity reward: reward = max - log(ppl) / scale
PPL_REWARD_MAX=6.0        # Maximum perplexity reward value; prev 6 for repeat penalty 6.0
REPETITION_PENALTIES=(6.0)  # Additive penalty: reward -= penalty * (1 - unique/total)

# Loss coefficients (grid)
VALUE_COEFS=(0.01)

# Logging
LOGGING_STEPS=1
SAVE_STEPS=20

# Grid definitions (edit these arrays to change the sweep)
LATENCY_COSTS=(100 0)
# LATENCY_COSTS=(100)
# ALLOWED_EXPERTS=(12 16) # training always collapse with 24 experts; possibly because the signals are too noisy?
ALLOWED_EXPERTS=(16)
LEARNING_RATES=(1e-3 2e-3)
# LEARNING_RATES=(4e-3)
PROMPTS_PER_STEP=(32)    # Total prompts per step (across all GPUs)
GRADIENT_ACCUMULATIONS=(2)  # Gradient accumulation steps to sweep over
SWITCH_INIT_BIASES=(-3)  # Initial switch head bias (0=50%, -3=5%, -4=2% switch rate)
CONTROLLER_HIDDEN_DIMS=(512)  # Hidden dimension for controller RNN (default: 4 * num_experts = 128)

# Advantage computation method
# ADVANTAGE_METHOD=ppo      # "ppo" = per-timestep V(s_t) baseline (original)
ADVANTAGE_METHOD=grpo       # "grpo" = group-level baseline (recommended)
NUM_GENERATIONS_PER_PROMPT=4  # Only used for GRPO: number of rollouts per prompt
# NOTE: For GRPO, per_device_batch must be divisible by NUM_GENERATIONS_PER_PROMPT
# The dataloader provides (per_device_batch / NUM_GENERATIONS_PER_PROMPT) unique prompts,
# then each prompt generates NUM_GENERATIONS_PER_PROMPT completions.
# This keeps wall-clock time the same as PPO with the same per_device_batch.

# Slurm resource settings (edit if you need different resources)
ACCOUNT=pli
PARTITION=pli-c
# ACCOUNT=henderson
# PARTITION=ailab
NODES=1
CPUS_PER_TASK=8
MEMORY=512G
GPUS_PER_JOB=4
TIME_LIMIT=72:00:00
# TIME_LIMIT=00:59:00


mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${LOG_DIR}"

run_idx=0
for latency in "${LATENCY_COSTS[@]}"; do
  for experts in "${ALLOWED_EXPERTS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for prompts in "${PROMPTS_PER_STEP[@]}"; do
        for grad_acc in "${GRADIENT_ACCUMULATIONS[@]}"; do
          for value_coef in "${VALUE_COEFS[@]}"; do
            for switch_bias in "${SWITCH_INIT_BIASES[@]}"; do
              for hidden_dim in "${CONTROLLER_HIDDEN_DIMS[@]}"; do
                for rep_penalty in "${REPETITION_PENALTIES[@]}"; do
          total_factor=$((GPUS_PER_JOB * grad_acc))
          if (( prompts % total_factor != 0 )); then
            echo "Skipping prompts=${prompts}, grad_acc=${grad_acc}: not divisible by world_size(${GPUS_PER_JOB}) * grad_acc(${grad_acc})"
            continue
          fi
          per_device_batch=$((prompts / total_factor))
          if (( per_device_batch < 1 )); then
            echo "Skipping prompts=${prompts}, grad_acc=${grad_acc}: per-device batch would be < 1"
            continue
          fi
          # For GRPO: check that per_device_batch is divisible by num_generations
          if [ "$ADVANTAGE_METHOD" = "grpo" ]; then
            if (( per_device_batch % NUM_GENERATIONS_PER_PROMPT != 0 )); then
              echo "Skipping prompts=${prompts}, grad_acc=${grad_acc}: per_device_batch(${per_device_batch}) not divisible by NUM_GENERATIONS_PER_PROMPT(${NUM_GENERATIONS_PER_PROMPT})"
              continue
            fi
          fi
          run_idx=$((run_idx + 1))
          # Include advantage method in job tag (grpo4 = grpo with 4 generations, ppo = ppo)
          if [ "$ADVANTAGE_METHOD" = "grpo" ]; then
            adv_tag="${ADVANTAGE_METHOD}${NUM_GENERATIONS_PER_PROMPT}"
          else
            adv_tag="${ADVANTAGE_METHOD}"
          fi
          job_tag="lat${latency}_exp${experts}_lr${lr}_pp${prompts}_ga${grad_acc}_vc${value_coef}_sb${switch_bias}_hd${hidden_dim}_rp${rep_penalty}_${adv_tag}"
          output_dir="${OUTPUT_ROOT}/${job_tag}"
          job_name="ctrl_${job_tag}"

          sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --gres=gpu:${GPUS_PER_JOB}
#SBATCH --constraint="gpu80"
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err

module load anaconda3/2024.2
module load proxy/default
conda activate ${ENV_PATH}

export PATH=${ENV_PATH}/bin:\$PATH
export HF_HOME=/scratch/gpfs/HENDERSON/zs7353/legacy/.cache/huggingface
export TORCH_HOME=/scratch/gpfs/HENDERSON/zs7353/legacy/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p ${output_dir}
cd ${BASE_DIR}

echo "Launching ${job_name} (run ${run_idx})"
accelerate launch \\
  --config_file ${BASE_DIR}/accelerate_config.yaml \\
  --mixed_precision=no \\
  train_controller_standalone.py \\
  --model-path ${MODEL_PATH} \\
  --data-dir ${DATA_DIR} \\
  --dataset-type ${DATASET_TYPE} \\
  --output-dir ${output_dir} \\
  --controller-allowed-experts ${experts} \\
  --prompts-per-category ${PROMPTS_PER_CATEGORY} \\
  --max-prompt-length ${MAX_PROMPT_LENGTH} \\
  --response-length ${RESPONSE_LENGTH} \\
  --learning-rate ${lr} \\
  --latency-cost ${latency} \\
  --per-device-train-batch ${per_device_batch} \\
  --gradient-accumulation ${grad_acc} \\
  --num-train-epochs ${NUM_TRAIN_EPOCHS} \\
  --num-update-epochs ${NUM_UPDATE_EPOCHS} \\
  --value-coef ${value_coef} \\
  --seed ${SEED} \\
  --ppl-reward-scale ${PPL_REWARD_SCALE} \\
  --ppl-reward-max ${PPL_REWARD_MAX} \\
  --repetition-penalty ${rep_penalty} \\
  --logging-steps ${LOGGING_STEPS} \\
  --save-steps ${SAVE_STEPS} \\
  --switch-init-bias ${switch_bias} \\
  --controller-hidden-dim ${hidden_dim} \\
  --advantage-method ${ADVANTAGE_METHOD} \\
  --num-generations-per-prompt ${NUM_GENERATIONS_PER_PROMPT}
EOF

                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Submitted ${run_idx} grid jobs."
