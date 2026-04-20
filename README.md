# Temporally Extended Mixture-of-Experts Models

This repository implements **temporally extended MoE** controller training using the **Option-Critic** framework with deliberation costs. A lightweight per-layer controller learns when to switch expert sets and which to load, reducing switch rates from >50% to under 5% while retaining up to ~90% of base-model accuracy.

**Paper:** [Temporally Extended Mixture-of-Experts Models](https://github.com/zeyushen-yo/rl_moe/blob/main/Temporally_extended_MoE_models___arxiv.pdf)

**Project Page:** [https://princeton-polaris-lab.github.io/moe_webpage/](https://princeton-polaris-lab.github.io/moe_webpage/)

## Features

- **Activation-based Controller**: Uses LLM hidden states directly with DeepSets for expert selection
- **Option-Critic with Deliberation Costs**: Learns when to switch expert sets via termination, value, and Plackett-Luce selection heads
- **Self-Distillation Reward**: Per-token reverse KL between frozen teacher and controller-augmented student

## Installation

### 1. Create Python Environment

```bash
conda create -n rl_moe python=3.11
conda activate rl_moe
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
# Edit launch_grid_activation.sh to configure hyperparameters
bash launch_grid_activation.sh
```

Or run directly via accelerate:

```bash
accelerate launch \
    --config_file accelerate_config.yaml \
    --mixed_precision=no \
    train_controller_standalone.py \
    --controller_type activation \
    --controller_allowed_experts 16 \
    --deliberation_cost 0.02 \
    --reward_type kl \
    --response-length 512 \
    --logging-steps 1 \
    --save-steps 20
```

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--controller_type` | Controller type (`activation`) | activation |
| `--controller_allowed_experts` | Number of experts in the mask ($k_{\text{mask}}$) | 16 |
| `--deliberation_cost` | Cost for switching experts ($\eta$) | 0.02 |
| `--reward_type` | Reward function (`kl`) | kl |
| `--with_lora` | Enable LoRA adapters on experts and attention | flag |

### Evaluation

Evaluation scripts are in the `eval/` directory:

```bash
# Evaluate controller on MATH
python eval/eval_math.py --checkpoint-dir /path/to/checkpoint --step 120 --controller-type activation

# Evaluate controller on MMLU/MMMLU
python eval/eval_mmlu_controller.py --checkpoint-dir /path/to/checkpoint --step 120 --controller-type activation

# Evaluate pruning baselines on MMLU/MMMLU
python eval/eval_mmlu_baseline.py --method frequency --num-experts 16
```

## Project Structure

```
rl_moe/
├── train_controller_standalone.py    # Main training entrypoint
├── activation_controller_trainer.py  # Activation-based controller trainer
├── launch_grid_activation.sh         # SLURM grid search launcher
├── run_controller.slurm              # Single-job SLURM script
├── accelerate_config.yaml            # Accelerate distributed config
├── deepspeed_config.json             # DeepSpeed config (referenced by accelerate)
├── transformers_patches/             # Custom transformers modifications
│   ├── models/gpt_oss/
│   │   ├── modeling_gpt_oss.py       # MoE model with controller hooks
│   │   ├── configuration_gpt_oss.py
│   │   └── __init__.py
│   ├── integrations/
│   │   └── mxfp4.py
│   ├── modeling_outputs.py
│   └── generation/
│       └── utils.py
├── eval/
│   ├── eval_math.py                  # MATH benchmark evaluation
│   ├── eval_mmlu_controller.py       # MMLU/MMMLU evaluation with controller
│   └── eval_mmlu_baseline.py         # MMLU/MMMLU evaluation for baselines
├── requirements.txt
├── install.sh                        # Patch installation script
└── README.md
```

## Citation

```bibtex
@article{shen2026temoe,
  title  = {Temporally Extended Mixture-of-Experts Models},
  author = {Shen, Zeyu and Henderson, Peter},
  year   = {2026}
}
```
