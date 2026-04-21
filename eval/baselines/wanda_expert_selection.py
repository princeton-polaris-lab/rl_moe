"""
Wanda-based expert selection for MoE models.

Based on "A Simple and Effective Pruning Approach for Large Language Models" (Sun et al., 2024)
https://arxiv.org/pdf/2306.11695

The Wanda metric scores weights by: S_ij = |W_ij| * ||X_j||_2
(weight magnitude × input activation norm)

For expert selection, we:
1. Run calibration data through the model
2. For each expert, compute Wanda scores for all its weights
3. Aggregate scores to get an "importance" score per expert
4. Keep the top-k experts per layer
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import re


class WandaExpertSelector:
    """
    Select experts based on Wanda importance scores.
    
    For each expert, we compute the sum of Wanda scores (|W| * ||X||_2) across all
    its weight parameters. Experts with higher total scores are more important.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        
        # Find MoE layers
        self.moe_layers = self._find_moe_layers()
        print(f"[WANDA] Found {len(self.moe_layers)} MoE layers")
        
        # Storage for activation norms
        self.input_norms: Dict[int, torch.Tensor] = {}  # layer_idx -> input activation norms
        self.hooks = []
        
    def _find_moe_layers(self) -> Dict[int, nn.Module]:
        """Find all MoE MLP modules with router and experts."""
        moe_layers = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'router') and hasattr(module, 'experts'):
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    moe_layers[layer_idx] = module
        return moe_layers
    
    def _register_hooks(self):
        """Register forward hooks to capture input activations to experts."""
        self.hooks = []
        self.input_norms = {}
        
        for layer_idx, moe_module in self.moe_layers.items():
            # We need to hook into the expert computation
            # The input to experts is the hidden state after router selection
            
            def make_hook(idx):
                def hook(module, inputs, outputs):
                    # inputs[0] is the hidden states going into the MoE block
                    if len(inputs) > 0:
                        hidden_states = inputs[0]
                        # Compute L2 norm across tokens for each hidden dimension
                        # Shape: (batch, seq_len, hidden_dim) -> (hidden_dim,)
                        if hidden_states.dim() == 3:
                            flat = hidden_states.view(-1, hidden_states.size(-1))
                        else:
                            flat = hidden_states
                        
                        norms = flat.norm(p=2, dim=0)  # (hidden_dim,)
                        
                        if idx not in self.input_norms:
                            self.input_norms[idx] = norms.clone()
                        else:
                            # Accumulate norms across batches
                            self.input_norms[idx] = (
                                self.input_norms[idx] ** 2 + norms ** 2
                            ).sqrt()
                return hook
            
            hook = moe_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def collect_activation_norms(
        self,
        sequences: List[torch.Tensor],
        batch_size: int = 4,
    ):
        """
        Collect input activation norms by running calibration data through the model.
        
        Args:
            sequences: List of tokenized sequences (each is a 1D tensor of token IDs)
            batch_size: Batch size for processing
        """
        self._register_hooks()
        self.model.eval()
        
        print(f"[WANDA] Collecting activation norms from {len(sequences)} sequences...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Collecting activations"):
                batch_seqs = sequences[i:i+batch_size]
                
                # Pad sequences to same length
                max_len = max(s.size(0) for s in batch_seqs)
                padded = torch.zeros(len(batch_seqs), max_len, dtype=torch.long, device=self.device)
                attention_mask = torch.zeros(len(batch_seqs), max_len, dtype=torch.long, device=self.device)
                
                for j, seq in enumerate(batch_seqs):
                    padded[j, :seq.size(0)] = seq
                    attention_mask[j, :seq.size(0)] = 1
                
                # Forward pass to collect activations
                _ = self.model(input_ids=padded, attention_mask=attention_mask)
        
        self._remove_hooks()
        print(f"[WANDA] Collected activation norms for {len(self.input_norms)} layers")
    
    def compute_expert_importance(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute Wanda importance scores for each expert in each layer.
        
        Returns:
            Dict mapping layer_idx -> list of (expert_idx, importance_score) tuples
            sorted by importance (highest first)
        """
        expert_importance = {}
        
        for layer_idx, moe_module in self.moe_layers.items():
            experts = moe_module.experts
            num_experts = experts.num_experts
            
            # Get input activation norms for this layer
            if layer_idx not in self.input_norms:
                print(f"Warning: No activation norms for layer {layer_idx}")
                continue
            
            input_norms = self.input_norms[layer_idx]  # (hidden_dim,)
            
            # Compute importance for each expert
            scores = []
            
            for exp_idx in range(num_experts):
                # GptOssExperts has fused parameters:
                # gate_up_proj: (num_experts, hidden_size, expert_dim*2)
                # down_proj: (num_experts, expert_dim, hidden_size)
                
                # Get weights for this expert
                gate_up_weight = experts.gate_up_proj[exp_idx]  # (hidden_size, expert_dim*2)
                down_weight = experts.down_proj[exp_idx]  # (expert_dim, hidden_size)
                
                # Compute Wanda score: |W| * ||X||_2
                # For gate_up_proj: input is hidden_states, shape (hidden_size,) norm
                # Score = sum over all weights of |W_ij| * ||X_j||_2
                
                # gate_up importance: weights are (hidden_size, expert_dim*2)
                # input norm is (hidden_size,)
                # Wanda score per weight: |W[i,j]| * input_norm[i]
                gate_up_score = (gate_up_weight.abs() * input_norms.unsqueeze(1)).sum()
                
                # For down_proj: input is the intermediate activation (expert_dim,)
                # We don't have exact norms for intermediate, so we use a proxy
                # Just use weight magnitude for down_proj (common simplification)
                down_score = down_weight.abs().sum()
                
                # Total expert importance
                total_score = (gate_up_score + down_score).item()
                scores.append((exp_idx, total_score))
            
            # Sort by importance (highest first)
            scores.sort(key=lambda x: x[1], reverse=True)
            expert_importance[layer_idx] = scores
        
        return expert_importance
    
    def get_selected_experts(
        self,
        num_experts_to_keep: int,
    ) -> Dict[int, List[int]]:
        """
        Get the top-k most important experts for each layer.
        
        Args:
            num_experts_to_keep: Number of experts to keep per layer
            
        Returns:
            Dict mapping layer_idx -> list of selected expert indices
        """
        importance = self.compute_expert_importance()
        
        selected = {}
        for layer_idx, scores in importance.items():
            # Take top-k experts
            top_experts = [exp_idx for exp_idx, score in scores[:num_experts_to_keep]]
            selected[layer_idx] = sorted(top_experts)
            
            print(f"[WANDA] Layer {layer_idx}: selected experts {top_experts[:8]}... "
                  f"(scores: {[f'{s:.1f}' for _, s in scores[:4]]}...)")
        
        return selected


def run_wanda_pruning(
    model: nn.Module,
    sequences: List[torch.Tensor],
    num_experts_to_keep: int,
    batch_size: int = 4,
) -> Dict[int, List[int]]:
    """
    Run Wanda-based expert selection.
    
    Args:
        model: The MoE model
        sequences: List of tokenized calibration sequences
        num_experts_to_keep: Number of experts to keep per layer
        batch_size: Batch size for processing
        
    Returns:
        Dict mapping layer_idx -> list of selected expert indices
    """
    selector = WandaExpertSelector(model)
    
    # Collect activation statistics
    selector.collect_activation_norms(sequences, batch_size)
    
    # Get selected experts
    selected = selector.get_selected_experts(num_experts_to_keep)
    
    return selected
