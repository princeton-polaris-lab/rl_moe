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
PROMPTS_PER_CATEGORY=1000 # for option-critic
# PROMPTS_PER_CATEGORY=500 # for grpo
MAX_PROMPT_LENGTH=1024
RESPONSE_LENGTH=1024
SEED=42

# Reward settings
# REWARD_TYPES=(ppl_only kl)  # ppl (perplexity + repetition), ppl_only (no repetition), kl (KL divergence)
REWARD_TYPES=(kl)
PPL_REWARD_SCALE=1.0      # Perplexity reward: reward = max - log(ppl) / scale
PPL_REWARD_MAX=6.0        # Maximum perplexity reward value; prev 6 for repeat penalty 6.0
REPETITION_PENALTIES=(6.0)  # Additive penalty: reward -= penalty * (1 - unique/total) (only for ppl type)
KL_REWARD_SCALE=1.0       # KL reward scale: reward = -kl_sum * scale

# Loss coefficients (grid)
VALUE_COEFS=(1)

# Logging
LOGGING_STEPS=1
SAVE_STEPS=20

# Grid definitions (edit these arrays to change the sweep)
LATENCY_COSTS=(0)
# LATENCY_COSTS=(0 100)
# ALLOWED_EXPERTS=(16 24)
ALLOWED_EXPERTS=(16)
# LEARNING_RATES=(2e-3)
LEARNING_RATES=(1e-3 2e-3)
# PROMPTS_PER_STEP=(32)    # Total prompts per step (across all GPUs), 32 for grpo, 16 for option critic
PROMPTS_PER_STEP=(16)
GRADIENT_ACCUMULATIONS=(2)  # Gradient accumulation steps to sweep over
SWITCH_INIT_BIASES=(3)  # Initial switch head bias
CONTROLLER_HIDDEN_DIMS=(512)  # Hidden dimension for controller RNN (default: 4 * num_experts = 128)
CONTROLLER_LAYER_NORMS=(1)  # LayerNorm on GRU hidden state: 1=on, 0=off (prevents gradient explosion)
CONTROLLER_INPUT_TYPES=(router_softmax)  # "router_softmax" (default) or "hidden_states" (richer but more compute/memory)
CONTROLLER_EXPERT_EMBED_DIMS=(128)  # DeepSets embedding dimension for expert set encoding in Q_U head
# CONTROLLER_INPUT_TYPES=(hidden_states)

# Controller type: "rnn" (GRU-based) or "activation" (LLM hidden states directly)
# CONTROLLER_TYPE=rnn
CONTROLLER_TYPE=activation  # "rnn" or "activation"
ACTIVATION_MLP_HIDDEN=512   # Hidden dim for activation controller MLPs (termination and Q heads)

# Advantage computation method
ADVANTAGE_METHOD=option_critic  # "option_critic" = Option-Critic (Harb et al. 2017, per-token TD)
# ADVANTAGE_METHOD=grpo             # "grpo" = group-level baseline (recommended for simplicity)
NUM_GENERATIONS_PER_PROMPT=4  # Only used for GRPO: number of rollouts per prompt
DELIBERATION_COSTS=(0.005)  # Option-Critic: η per switch (grid over values), 0.001 for rnn, 0.01 for activation
GAMMA=1  # Discount factor for Option-Critic TD (1 = all tokens equally weighted)
GAE_LAMBDA=0.95  # GAE lambda for bias-variance tradeoff (0=TD(0), 1=MC)
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
                for expert_embed_dim in "${CONTROLLER_EXPERT_EMBED_DIMS[@]}"; do
                  for layer_norm in "${CONTROLLER_LAYER_NORMS[@]}"; do
                    for input_type in "${CONTROLLER_INPUT_TYPES[@]}"; do
                      for reward_type in "${REWARD_TYPES[@]}"; do
                        for delib_cost in "${DELIBERATION_COSTS[@]}"; do
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
          
          # Determine repetition penalty based on reward type
          # Only ppl uses repetition penalty; ppl_only and kl do not
          if [ "$reward_type" = "ppl" ]; then
            rep_penalty=${REPETITION_PENALTIES[0]}
            rep_tag="rp${rep_penalty}"
          else
            rep_penalty=0.0
            rep_tag=""  # No rep penalty tag for non-ppl types
          fi
          
          run_idx=$((run_idx + 1))
          # Include advantage method in job tag (grpo4 = grpo with 4 generations, ppo = ppo)
          if [ "$ADVANTAGE_METHOD" = "grpo" ]; then
            adv_tag="${ADVANTAGE_METHOD}${NUM_GENERATIONS_PER_PROMPT}"
          else
            adv_tag="${ADVANTAGE_METHOD}"
          fi
          # Include layer_norm in tag: ln1 or ln0
          ln_tag="ln${layer_norm}"
          # Include input_type in tag: rs (router_softmax) or hs (hidden_states)
          if [ "$input_type" = "hidden_states" ]; then
            it_tag="hs"
          else
            it_tag="rs"
          fi
          # Include reward type in tag
          rwd_tag="rwd_${reward_type}"
          # Include deliberation cost in tag (dc0, dc0.1, etc)
          dc_tag="dc${delib_cost}"
          # Include controller type in tag: rnn or act (activation)
          if [ "$CONTROLLER_TYPE" = "activation" ]; then
            ctrl_tag="act"
          else
            ctrl_tag="rnn"
          fi
          
          # Build job tag - include rep_tag only if non-empty
          if [ -n "$rep_tag" ]; then
            job_tag="lat${latency}_exp${experts}_lr${lr}_pp${prompts}_ga${grad_acc}_vc${value_coef}_sb${switch_bias}_hd${hidden_dim}_${rep_tag}_${ln_tag}_${it_tag}_${rwd_tag}_${dc_tag}_${ctrl_tag}_${adv_tag}"
          else
            job_tag="lat${latency}_exp${experts}_lr${lr}_pp${prompts}_ga${grad_acc}_vc${value_coef}_sb${switch_bias}_hd${hidden_dim}_${ln_tag}_${it_tag}_${rwd_tag}_${dc_tag}_${ctrl_tag}_${adv_tag}"
          fi
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
  --reward-type ${reward_type} \\
  --ppl-reward-scale ${PPL_REWARD_SCALE} \\
  --ppl-reward-max ${PPL_REWARD_MAX} \\
  --repetition-penalty ${rep_penalty} \\
  --kl-reward-scale ${KL_REWARD_SCALE} \\
  --logging-steps ${LOGGING_STEPS} \\
  --save-steps ${SAVE_STEPS} \\
  --switch-init-bias ${switch_bias} \\
  --controller-hidden-dim ${hidden_dim} \\
  --controller-expert-embed-dim ${expert_embed_dim} \\
  --controller-layer-norm ${layer_norm} \\
  --controller-input-type ${input_type} \\
  --advantage-method ${ADVANTAGE_METHOD} \\
  --num-generations-per-prompt ${NUM_GENERATIONS_PER_PROMPT} \\
  --deliberation-cost ${delib_cost} \\
  --gamma ${GAMMA} \\
  --gae-lambda ${GAE_LAMBDA} \\
  --controller-type ${CONTROLLER_TYPE} \\
  --activation-controller-mlp-hidden ${ACTIVATION_MLP_HIDDEN}
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
      done
    done
  done
done

echo "Submitted ${run_idx} grid jobs."
