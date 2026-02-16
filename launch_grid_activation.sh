#!/bin/bash

# Root paths (edit these if you move the project or change model/data locations)
BASE_DIR=/scratch/gpfs/HENDERSON/zs7353/rl_moe
ENV_PATH=/scratch/gpfs/HENDERSON/zs7353/envs/trl
MODEL_PATH=/scratch/gpfs/KOROLOVA/gpt-oss-20b
# Data paths - update these based on DATASET_TYPE
DATASET_TYPE=math  # 'mmlu' for MMLU multiple-choice, 'nemotron' for open-ended, 'math' for Hendrycks MATH
if [ "$DATASET_TYPE" = "mmlu" ]; then
    DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/mmlu
elif [ "$DATASET_TYPE" = "math" ]; then
    DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/rl_moe/hendrycks_math
else
    DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/Nemotron-Post-Training-Dataset-v2/data
fi
MATH_DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/rl_moe/hendrycks_math
MATH_SPLIT=train  # 'train' for training, 'test' for eval
# Category filter for Nemotron dataset (comma-separated, e.g., "math" or "math,code")
# Set to empty string "" to use all categories
DATA_CATEGORIES="math"
# DATA_CATEGORIES=""
OUTPUT_ROOT=${BASE_DIR}/controller_rl_grid
LOG_DIR=${BASE_DIR}/logs

# Training hyper-parameters that stay constant across the sweep
NUM_UPDATE_EPOCHS=1       # PPO epochs per batch (1 = fully on-policy)
NUM_TRAIN_EPOCHS=1        # Total training epochs
PROMPTS_PER_CATEGORY=10000 # for math only
# PROMPTS_PER_CATEGORY=2000
MAX_PROMPT_LENGTH=512
# with correctness reward, we need the rollouts in one batch to come from the same prompt
NUM_ROLLOUTS_PER_PROMPT=0  # Number of rollouts per prompt (1=different prompts, 4=same prompt x4 rollouts)
# NUM_ROLLOUTS_PER_PROMPT=1
# Response length for generation (grid-searchable)
# RESPONSE_LENGTHS=(1024)
RESPONSE_LENGTHS=(512)  # To compare different response lengths
TOKEN_TEMPERATURE=0.5  # Temperature for token sampling (lower = more deterministic)
# TOKEN_TEMPERATURE=0 # no intra-option policy update
SEED=42

# Reward settings
KL_REWARD_SCALE=1.0       # KL reward scale: reward = -kl_sum * scale
# CORRECTNESS_REWARD_ALPHAS=(0.1 1)  # Per-token bonus for correct trajectories (grid searchable)
CORRECTNESS_REWARD_ALPHAS=(0)  # To compare with/without correctness reward

# Loss coefficients (grid)
# VALUE_COEFS=(0.1 1)
VALUE_COEFS=(0.01)
ENTROPY_COEFS=(0)  # Entropy bonus: 0 = disabled, try 0.01-0.1 to encourage exploration
# ENTROPY_COEFS=(0 0.01)

# Logging
LOGGING_STEPS=1
SAVE_STEPS=20

# Resume from checkpoint (empty string = start fresh, or path to .pt file, or "latest")
# RESUME_CHECKPOINT="/scratch/gpfs/HENDERSON/zs7353/rl_moe/controller_rl_grid/lat0_exp16_lr1e-3_pp32_ga2_vc0.01_sb-3_hd512_ln1_rwd_kl_grpo4/controller_step_380.pt"
RESUME_CHECKPOINT=""

# Grid definitions (edit these arrays to change the sweep)
LATENCY_COSTS=(0)
# LATENCY_COSTS=(100)
ALLOWED_EXPERTS=(16)
# ALLOWED_EXPERTS=(16)
LEARNING_RATES=(1e-4)
# LEARNING_RATES=(1e-3 1e-4)
# PROMPTS_PER_STEP=(32)    # Total prompts per step (across all GPUs), 32 for grpo, 16 for option critic
PROMPTS_PER_STEP=(16)
GRADIENT_ACCUMULATIONS=(1)  # Gradient accumulation steps to sweep over
SWITCH_INIT_BIASES=(-3)  # Initial switch head bias
CONTROLLER_HIDDEN_DIMS=(512)  # Hidden dimension for controller RNN (default: 4 * num_experts = 128)
CONTROLLER_LAYER_NORMS=(1)  # LayerNorm on GRU hidden state: 1=on, 0=off (prevents gradient explosion)
CONTROLLER_INPUT_TYPES=(router_softmax)  # "router_softmax" (default) or "hidden_states" (richer but more compute/memory)
CONTROLLER_EXPERT_EMBED_DIMS=(128)  # DeepSets embedding dimension for expert set encoding in Q_U head
# CONTROLLER_INPUT_TYPES=(hidden_states)

# Controller type: "rnn" (GRU-based) or "activation" (LLM hidden states directly)
# CONTROLLER_TYPE=rnn
CONTROLLER_TYPE=activation  # "rnn" or "activation". Remember to implement intra-option update for activation controller as well
ACTIVATION_MLP_HIDDEN=1024   # Hidden dim for activation controller MLPs (termination and Q heads)

# Option-Critic settings
DELIBERATION_COSTS=(0.05)  # activation based controller
GAMMAS=(0.95)  # Discount factor for TD targets (grid searchable)
# GAMMAS=(0.9)
GAE_LAMBDA=0.95  # GAE lambda for bias-variance tradeoff (0=TD(0), 1=MC)

# Intra-option policy update (Harb et al. 2017, Algorithm 1)
# When enabled, updates router + experts via policy gradient on token predictions
INTRA_OPTION_UPDATE=1  # 1=enable, 0=disable (only applies to option_critic)
# INTRA_OPTION_UPDATE=0
INTRA_OPTION_LRS=(2e-4)   # Learning rate for LLM (LoRA + router) params (grid searchable)
# INTRA_OPTION_LRS=(1e-4 5e-4)
INTRA_OPTION_WARMUP_STEPS=0  # Skip LLM updates for first N steps (let value function warm up)
INTRA_OPTION_Q_BASELINES=(0)  # Use Q(s,o) baseline for intra-option gradient (A2OC: G-Q). 0=off, 1=on
# INTRA_OPTION_Q_BASELINES=(1 0)  # To compare with and without Q baseline
LORA_RS=(16)            # LoRA rank for expert adapters (grid searchable)
# LORA_RS=(4 8 16)
LORA_ALPHA=16          # LoRA alpha scaling factor

# Termination advantage RMS normalization
# Helps when advantage variance collapses during training
TERM_ADV_RMS_NORMS=(1)  # 0=off, 1=on (grid searchable)
# TERM_ADV_RMS_NORMS=(0 1)

# TopK termination regularization
# Loss = λ * (1 - mean(TopK β))²
# Prevents termination head from collapsing to uniform-low
TERM_TOPK_LAMBDAS=(0.0)  # Grid-searchable: e.g., (0.0 0.1 1.0)
# TERM_TOPK_LAMBDAS=(0.0 0.1 0.5 1.0)
TERM_TOPK_KS=(1000)  # Grid-searchable: e.g., (100 500 1000)
# TERM_TOPK_KS=(500 1000 2000)

# Q-based expert selection (alternative to Plackett-Luce)
# Selects options via argmax Q instead of learned Plackett-Luce policy
# When enabled, no selection policy gradient is computed - only Q is trained via TD
Q_BASED_SELECTIONS=(1)  # 0=Plackett-Luce, 1=Q-based (grid searchable)
# Q_BASED_SELECTIONS=(0 1)  # To compare PL vs Q-based
Q_SELECTION_STEPS=10      # Number of gradient ascent steps for Q-based
Q_SELECTION_LR=1.0        # Learning rate for Q-based optimization
Q_SELECTION_EPSILONS=(0)  # Fixed ε for Q-based selection (if not using annealing)
# Q_SELECTION_EPSILONS=(0.0 0.1 0.2)  # To compare different exploration rates
# Annealing schedule for Q-based selection epsilon
Q_SELECTION_EPSILON_START=1  # Starting ε (set to "" to disable annealing)
Q_SELECTION_EPSILON_END=0.1   # Final ε after annealing
Q_SELECTION_EPSILON_ANNEAL_STEPS=200  # Steps to anneal over
Q_SELECTION_DEBUG=0       # 1=print debug info for Q-based selection
Q_SELECTION_INIT_WS=(1.0) # Initial weight for current experts in Q-based selection (grid searchable)
# Q_SELECTION_INIT_WS=(0.0 1.0)  # Higher = more concentrated on current option initially

# Exploration: ε-greedy mixture for expert selection (Plackett-Luce mode)
# π_mixed = ε × Uniform + (1-ε) × Policy
# Provides guaranteed exploration without breaking gradient computation
# NOTE: Only used when Q_BASED_SELECTION=0 (Plackett-Luce)
SELECTION_EPSILONS=(0)  # Fixed epsilon (if not using annealing)
# Annealing schedule for Plackett-Luce epsilon (leave empty to disable)
SELECTION_EPSILON_START=""  # e.g., "0.5" to enable annealing from 0.5
SELECTION_EPSILON_END=0.05
SELECTION_EPSILON_ANNEAL_STEPS=200

# Repetition penalty (distance-based)
# For each token, penalty = c * λ^d where d is distance to previous occurrence
# c should be negative for penalty, λ < 1 so nearby repeats penalized more
REPETITION_PENALTY_CS=(0)
# REPETITION_PENALTY_CS=(-1)
# REPETITION_PENALTY_DECAYS=(0.95 0.99)  # Grid-searchable: e.g., (0.8 0.9 0.95)
REPETITION_PENALTY_DECAYS=(0.99)
# REPETITION_PENALTY_DECAYS=(0)

# Teacher-mixed sampling (MiniLLM-style)
# During rollout: p_mixed = α * p_teacher + (1-α) * p_student
# Helps prevent reward hacking by mixing in teacher distribution
# α = 0.2 is the MiniLLM default (https://arxiv.org/pdf/2306.08543)
TEACHER_MIX_ALPHAS=(0.2)  # Grid-searchable: e.g., (0.0 0.1 0.2)
# TEACHER_MIX_ALPHAS=(0.0 0.2)  # To compare with/without teacher mixing

# Slurm resource settings (edit if you need different resources)
# ACCOUNT=pli
# PARTITION=pli-c
ACCOUNT=henderson
PARTITION=ailab
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
            for entropy_coef in "${ENTROPY_COEFS[@]}"; do
            for switch_bias in "${SWITCH_INIT_BIASES[@]}"; do
              for hidden_dim in "${CONTROLLER_HIDDEN_DIMS[@]}"; do
                for expert_embed_dim in "${CONTROLLER_EXPERT_EMBED_DIMS[@]}"; do
                  for layer_norm in "${CONTROLLER_LAYER_NORMS[@]}"; do
                    for input_type in "${CONTROLLER_INPUT_TYPES[@]}"; do
                      for delib_cost in "${DELIBERATION_COSTS[@]}"; do
                        for gamma in "${GAMMAS[@]}"; do
                          for intra_lr in "${INTRA_OPTION_LRS[@]}"; do
                            for sel_eps in "${SELECTION_EPSILONS[@]}"; do
                              for lora_r in "${LORA_RS[@]}"; do
                                for term_rms in "${TERM_ADV_RMS_NORMS[@]}"; do
                                  for q_based in "${Q_BASED_SELECTIONS[@]}"; do
                                    for q_init_w in "${Q_SELECTION_INIT_WS[@]}"; do
                                      for q_sel_eps in "${Q_SELECTION_EPSILONS[@]}"; do
                                        for rep_c in "${REPETITION_PENALTY_CS[@]}"; do
                                          for rep_decay in "${REPETITION_PENALTY_DECAYS[@]}"; do
                                            for term_topk_lambda in "${TERM_TOPK_LAMBDAS[@]}"; do
                                              for term_topk_k in "${TERM_TOPK_KS[@]}"; do
                                                for resp_len in "${RESPONSE_LENGTHS[@]}"; do
                                                  for q_baseline in "${INTRA_OPTION_Q_BASELINES[@]}"; do
                                                    for corr_alpha in "${CORRECTNESS_REWARD_ALPHAS[@]}"; do
                                                      for teach_alpha in "${TEACHER_MIX_ALPHAS[@]}"; do
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
          
          run_idx=$((run_idx + 1))
          
          # Include layer_norm in tag: ln1 or ln0
          ln_tag="ln${layer_norm}"
          # Include input_type in tag: rs (router_softmax) or hs (hidden_states)
          if [ "$input_type" = "hidden_states" ]; then
            it_tag="hs"
          else
            it_tag="rs"
          fi
          # Include deliberation cost in tag (dc0, dc0.1, etc)
          dc_tag="dc${delib_cost}"
          # Include gamma in tag (g0.9, g0.99, etc)
          gamma_tag="g${gamma}"
          # Include controller type in tag: rnn or act (activation)
          if [ "$CONTROLLER_TYPE" = "activation" ]; then
            ctrl_tag="act"
          else
            ctrl_tag="rnn"
          fi
          
          # Include intra-option update params in tag (only when enabled)
          if [ "$INTRA_OPTION_UPDATE" = "1" ]; then
            intra_tag="intra_lr${intra_lr}_r${lora_r}_a${LORA_ALPHA}"
          else
            intra_tag=""
          fi
          
          # Include term_adv_rms_norm in tag (only if enabled)
          if [ "$term_rms" = "1" ]; then
            term_rms_tag="trms"
          else
            term_rms_tag=""
          fi
          
          # Include Q-based selection in tag (only if enabled)
          if [ "$q_based" = "1" ]; then
            qsel_tag="qsel_w${q_init_w}_e${q_sel_eps}"
          else
            qsel_tag=""
          fi
          
          # Build annealing arguments for Plackett-Luce selection epsilon
          if [ -n "$SELECTION_EPSILON_START" ]; then
            SELECTION_EPSILON_ANNEAL_ARGS="--selection-epsilon-start ${SELECTION_EPSILON_START} --selection-epsilon-end ${SELECTION_EPSILON_END} --selection-epsilon-anneal-steps ${SELECTION_EPSILON_ANNEAL_STEPS}"
          else
            SELECTION_EPSILON_ANNEAL_ARGS=""
          fi
          
          # Build annealing arguments for Q-based selection epsilon
          if [ -n "$Q_SELECTION_EPSILON_START" ]; then
            Q_SELECTION_EPSILON_ANNEAL_ARGS="--q-selection-epsilon-start ${Q_SELECTION_EPSILON_START} --q-selection-epsilon-end ${Q_SELECTION_EPSILON_END} --q-selection-epsilon-anneal-steps ${Q_SELECTION_EPSILON_ANNEAL_STEPS}"
            q_eps_anneal_tag="qea${Q_SELECTION_EPSILON_START}-${Q_SELECTION_EPSILON_END}_${Q_SELECTION_EPSILON_ANNEAL_STEPS}"
          else
            Q_SELECTION_EPSILON_ANNEAL_ARGS=""
            q_eps_anneal_tag=""
          fi
          
          # Build annealing tag for Plackett-Luce selection epsilon
          if [ -n "$SELECTION_EPSILON_START" ]; then
            pl_eps_anneal_tag="plea${SELECTION_EPSILON_START}-${SELECTION_EPSILON_END}_${SELECTION_EPSILON_ANNEAL_STEPS}"
          else
            pl_eps_anneal_tag=""
          fi
          
          # Include selection epsilon in tag (only if non-zero)
          if [ "$sel_eps" != "0.0" ] && [ "$sel_eps" != "0" ]; then
            eps_tag="eps${sel_eps}"
          else
            eps_tag=""
          fi
          
          # Build entropy tag (only if non-zero)
          if [ "$entropy_coef" != "0" ]; then
            ent_tag="ec${entropy_coef}"
          else
            ent_tag=""
          fi
          
          # Build repetition penalty tag (only if c != 0)
          if [ "$rep_c" != "0.0" ] && [ "$rep_c" != "0" ]; then
            rep_tag="rep_c${rep_c}_d${rep_decay}"
          else
            rep_tag=""
          fi
          
          # Build TopK termination regularization tag (only if lambda != 0)
          if [ "$term_topk_lambda" != "0.0" ] && [ "$term_topk_lambda" != "0" ]; then
            topk_tag="topk_l${term_topk_lambda}_k${term_topk_k}"
          else
            topk_tag=""
          fi
          
          # Response length tag
          rl_tag="rl${resp_len}"
          
          # Q baseline tag (only if enabled)
          if [ "$q_baseline" = "1" ]; then
            qb_tag="qb"
          else
            qb_tag=""
          fi
          
          # Correctness reward tag (only if non-zero)
          if [ "$corr_alpha" != "0.0" ] && [ "$corr_alpha" != "0" ]; then
            corr_tag="corr${corr_alpha}"
          else
            corr_tag=""
          fi
          
          # Teacher mixing tag (only if non-zero)
          if [ "$teach_alpha" != "0.0" ] && [ "$teach_alpha" != "0" ]; then
            teach_tag="tmix${teach_alpha}"
          else
            teach_tag=""
          fi
          
          # Build job tag - include optional tags only if non-empty
          # Base tag
          base_tag="lat${latency}_exp${experts}_lr${lr}_pp${prompts}_ga${grad_acc}_vc${value_coef}_sb${switch_bias}_hd${hidden_dim}"
          # Add optional ent_tag
          [ -n "$ent_tag" ] && base_tag="${base_tag}_${ent_tag}"
          # Add fixed tags
          base_tag="${base_tag}_${ln_tag}_${it_tag}_${dc_tag}_${gamma_tag}_${ctrl_tag}"
          # Add optional intra_tag (intra-option update params)
          [ -n "$intra_tag" ] && base_tag="${base_tag}_${intra_tag}"
          # Add optional eps_tag (selection epsilon for exploration)
          [ -n "$eps_tag" ] && base_tag="${base_tag}_${eps_tag}"
          # Add optional term_rms_tag (termination advantage RMS normalization)
          [ -n "$term_rms_tag" ] && base_tag="${base_tag}_${term_rms_tag}"
          # Add optional qsel_tag (Q-based selection)
          [ -n "$qsel_tag" ] && base_tag="${base_tag}_${qsel_tag}"
          # Add optional Q-based epsilon annealing tag
          [ -n "$q_eps_anneal_tag" ] && base_tag="${base_tag}_${q_eps_anneal_tag}"
          # Add optional PL epsilon annealing tag
          [ -n "$pl_eps_anneal_tag" ] && base_tag="${base_tag}_${pl_eps_anneal_tag}"
          # Add optional rep_tag (repetition penalty)
          [ -n "$rep_tag" ] && base_tag="${base_tag}_${rep_tag}"
          # Add optional topk_tag (TopK termination regularization)
          [ -n "$topk_tag" ] && base_tag="${base_tag}_${topk_tag}"
          # Add response length tag
          base_tag="${base_tag}_${rl_tag}"
          # Add optional qb_tag (Q baseline for intra-option)
          [ -n "$qb_tag" ] && base_tag="${base_tag}_${qb_tag}"
          # Add optional corr_tag (correctness reward)
          [ -n "$corr_tag" ] && base_tag="${base_tag}_${corr_tag}"
          # Add optional teach_tag (teacher mixing)
          [ -n "$teach_tag" ] && base_tag="${base_tag}_${teach_tag}"
          # Add data category tag if filtering (for nemotron dataset)
          if [ -n "$DATA_CATEGORIES" ]; then
            cat_tag="cat_${DATA_CATEGORIES//,/_}"  # Replace commas with underscores
            base_tag="${base_tag}_${cat_tag}"
          fi
          job_tag="${base_tag}"
          output_dir="${OUTPUT_ROOT}/${job_tag}"
          job_name="ctrl_${job_tag}"
          
          # Build resume flag (empty if not resuming)
          if [ -n "$RESUME_CHECKPOINT" ]; then
            RESUME_FLAG="--resume-from-checkpoint ${RESUME_CHECKPOINT}"
          else
            RESUME_FLAG=""
          fi

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
  --data-categories "${DATA_CATEGORIES}" \\
  --math-data-dir ${MATH_DATA_DIR} \\
  --math-split ${MATH_SPLIT} \\
  --correctness-reward-alpha ${corr_alpha} \\
  --output-dir ${output_dir} \\
  --controller-allowed-experts ${experts} \\
  --prompts-per-category ${PROMPTS_PER_CATEGORY} \\
  --max-prompt-length ${MAX_PROMPT_LENGTH} \\
  --num-rollouts-per-prompt ${NUM_ROLLOUTS_PER_PROMPT} \\
  --response-length ${resp_len} \\
  --token-temperature ${TOKEN_TEMPERATURE} \\
  --learning-rate ${lr} \\
  --latency-cost ${latency} \\
  --per-device-train-batch ${per_device_batch} \\
  --gradient-accumulation ${grad_acc} \\
  --num-train-epochs ${NUM_TRAIN_EPOCHS} \\
  --num-update-epochs ${NUM_UPDATE_EPOCHS} \\
  --value-coef ${value_coef} \\
  --entropy-coef ${entropy_coef} \\
  --seed ${SEED} \\
  --kl-reward-scale ${KL_REWARD_SCALE} \\
  --logging-steps ${LOGGING_STEPS} \\
  --save-steps ${SAVE_STEPS} \\
  --switch-init-bias ${switch_bias} \\
  --controller-hidden-dim ${hidden_dim} \\
  --controller-expert-embed-dim ${expert_embed_dim} \\
  --controller-layer-norm ${layer_norm} \\
  --controller-input-type ${input_type} \\
  --deliberation-cost ${delib_cost} \\
  --gamma ${gamma} \\
  --gae-lambda ${GAE_LAMBDA} \\
  --controller-type ${CONTROLLER_TYPE} \\
  --activation-controller-mlp-hidden ${ACTIVATION_MLP_HIDDEN} \\
  --intra-option-update ${INTRA_OPTION_UPDATE} \\
  --intra-option-lr ${intra_lr} \\
  --intra-option-warmup-steps ${INTRA_OPTION_WARMUP_STEPS} \\
  --intra-option-q-baseline ${q_baseline} \\
  --lora-r ${lora_r} \\
  --lora-alpha ${LORA_ALPHA} \\
  --selection-epsilon ${sel_eps} \\
  ${SELECTION_EPSILON_ANNEAL_ARGS} \\
  --term-adv-rms-norm ${term_rms} \\
  --q-based-selection ${q_based} \\
  --q-selection-steps ${Q_SELECTION_STEPS} \\
  --q-selection-lr ${Q_SELECTION_LR} \\
  --q-selection-epsilon ${q_sel_eps} \\
  ${Q_SELECTION_EPSILON_ANNEAL_ARGS} \\
  --q-selection-debug ${Q_SELECTION_DEBUG} \\
  --q-selection-init-w ${q_init_w} \\
  --repetition-penalty-c ${rep_c} \\
  --repetition-penalty-decay ${rep_decay} \\
  --term-topk-lambda ${term_topk_lambda} \\
  --term-topk-k ${term_topk_k} \\
  --teacher-mix-alpha ${teach_alpha} \\
  ${RESUME_FLAG}
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
    done
  done
done

echo "Submitted ${run_idx} grid jobs."
