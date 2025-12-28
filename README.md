# RL-MoE: Reinforcement Learning for Mixture-of-Experts Controller Training

This repository implements controller training for Mixture-of-Experts (MoE) models using reinforcement learning, specifically the **Option-Critic** algorithm.

## Features

- **RNN-based Controller**: Uses a GRU to maintain hidden state across tokens for expert selection
- **Activation-based Controller**: Uses LLM hidden states directly with DeepSets for expert selection
- **Multiple Reward Functions**: KL divergence, perplexity-based rewards
- **Distributed Training**: Support for DeepSpeed and accelerate

## Installation

### 1. Create Python Environment

```bash
# Using conda
conda create -n rl_moe python=3.11
conda activate rl_moe

# Or using venv
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Apply Transformers Patches

This project requires custom modifications to the `transformers` library for MoE controller support.

```bash
chmod +x install.sh
./install.sh
```

Or specify a custom environment path:

```bash
./install.sh /path/to/your/python/env
```

## Usage

### Training a Controller

Use the launch script for grid search:

```bash
# Edit launch_grid.sh to configure hyperparameters
bash launch_grid.sh
```

Or run directly:

```bash
python train_controller_standalone.py \
    --model_name /path/to/gpt-oss-20b \
    --learning_rate 1e-3 \
    --per_prompt_generation 16 \
    --grad_accum_steps 2 \
    --controller_allowed_experts 16 \
    --reward_type kl \
    --algorithm option_critic \
    --controller_type rnn  # or "activation"
```

### Controller Types

- **`rnn`**: RNN-based controller (GRU hidden state)
- **`activation`**: Activation-based controller (uses LLM hidden states)

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--learning_rate` | Controller learning rate | 1e-3 |
| `--per_prompt_generation` | Batch size per prompt | 16 |
| `--grad_accum_steps` | Gradient accumulation steps | 2 |
| `--controller_allowed_experts` | Number of experts to allow | 16 |
| `--switch_init_bias` | Initial bias for switch probability | -3.0 |
| `--deliberation_cost` | Cost for switching experts | 0.001 |
| `--reward_type` | Reward function (kl, perplexity) | kl |

## Project Structure

```
rl_moe/
├── train_controller_standalone.py  # Main training script
├── controller_trainer.py           # RNN controller trainer
├── activation_controller_trainer.py # Activation controller trainer
├── launch_grid.sh                  # SLURM job submission script
├── transformers_patches/           # Custom transformers modifications
│   ├── models/gpt_oss/
│   │   ├── modeling_gpt_oss.py     # MoE model with controller support
│   │   └── configuration_gpt_oss.py
│   └── integrations/
│       └── mxfp4.py                # Optimized MoE kernel
├── requirements.txt
├── install.sh                      # Patch installation script
└── README.md
```