"""
Reconstruction loss-based expert pruning with greedy expert addition.

Following NAEE paper (https://arxiv.org/pdf/2402.14800):
- For each layer, find subset of experts that minimizes reconstruction loss
- Reconstruction loss = ||F(x) - F'(x, C)||_F (Frobenius norm)
- F(x) is original layer output, F'(x, C) is output with only subset C of experts

Original NAEE uses exhaustive enumeration (feasible for 8 experts).
For 32 experts -> 16, exhaustive enumeration is infeasible (C(32,16) = 601M combinations).
So we use GREEDY ADDITION: start with empty set, iteratively add best expert.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm


class ReconstructionExpertSelector:
    """
    Selects experts by minimizing reconstruction loss using greedy addition.
    """
    
    def __init__(self, model, num_experts_to_keep: int = 16):
        """
        Args:
            model: The MoE model
            num_experts_to_keep: Number of experts to keep per layer
        """
        self.model = model
        self.num_experts_to_keep = num_experts_to_keep
        
        # Find MoE layers and their routers
        self.moe_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'router') and hasattr(module, 'experts'):
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    self.moe_layers.append((layer_idx, name, module))
        
        self.moe_layers.sort(key=lambda x: x[0])
        
        if self.moe_layers:
            first_router = self.moe_layers[0][2].router
            self.num_total_experts = first_router.weight.shape[0]
            self.top_k = getattr(first_router, 'top_k', 8)
        else:
            self.num_total_experts = 32
            self.top_k = 8
        
        print(f"[ReconstructionSelector] Found {len(self.moe_layers)} MoE layers")
        print(f"[ReconstructionSelector] {self.num_total_experts} experts, top-{self.top_k} routing")
        print(f"[ReconstructionSelector] Will keep {num_experts_to_keep} experts per layer")
    
    def cache_layer_io(
        self,
        sequences: List[torch.Tensor],
        batch_size: int = 4,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Cache input-output pairs for each MoE layer.
        
        Args:
            sequences: Calibration sequences
            batch_size: Batch size
            
        Returns:
            Dict mapping layer_idx -> (inputs, outputs)
            Each tensor has shape (total_tokens, hidden_dim)
        """
        self.model.eval()
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        device = next(self.model.parameters()).device
        
        # Storage for inputs and outputs per layer
        layer_inputs: Dict[int, List[torch.Tensor]] = defaultdict(list)
        layer_outputs: Dict[int, List[torch.Tensor]] = defaultdict(list)
        
        # Install hooks to capture inputs and outputs
        hooks = []
        
        def make_input_hook(layer_idx):
            def hook(module, args):
                if len(args) == 0:
                    return
                hidden_states = args[0]
                # Flatten to (num_tokens, hidden_dim)
                if len(hidden_states.shape) == 3:
                    hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
                else:
                    hidden_flat = hidden_states
                layer_inputs[layer_idx].append(hidden_flat.detach().cpu())
            return hook
        
        def make_output_hook(layer_idx):
            def hook(module, args, output):
                # Output is typically a tuple, get the hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                # Flatten to (num_tokens, hidden_dim)
                if len(hidden_states.shape) == 3:
                    hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
                else:
                    hidden_flat = hidden_states
                layer_outputs[layer_idx].append(hidden_flat.detach().cpu())
            return hook
        
        # Install hooks
        for layer_idx, name, mlp_module in self.moe_layers:
            pre_hook = mlp_module.register_forward_pre_hook(make_input_hook(layer_idx))
            post_hook = mlp_module.register_forward_hook(make_output_hook(layer_idx))
            hooks.extend([pre_hook, post_hook])
        
        print(f"[ReconstructionSelector] Caching layer I/O for {len(sequences)} sequences...")
        
        # Forward pass through all sequences
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Caching layer I/O"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(sequences))
            batch_seqs = sequences[start:end]
            
            input_ids = torch.stack(batch_seqs).to(device)
            
            with torch.no_grad():
                _ = self.model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        self.model.config.use_cache = use_cache
        torch.cuda.empty_cache()
        
        # Concatenate all cached tensors
        cached = {}
        for layer_idx in layer_inputs:
            inputs_cat = torch.cat(layer_inputs[layer_idx], dim=0)
            outputs_cat = torch.cat(layer_outputs[layer_idx], dim=0)
            cached[layer_idx] = (inputs_cat, outputs_cat)
            print(f"  Layer {layer_idx}: {inputs_cat.shape[0]} tokens cached")
        
        return cached
    
    def compute_single_expert_output(
        self,
        experts_module,
        hidden_states: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        """
        Compute output for a single expert using fused GptOssExperts parameters.
        
        GptOssExperts stores parameters as:
          - gate_up_proj: (num_experts, hidden_size, expert_dim*2)
          - gate_up_proj_bias: (num_experts, expert_dim*2)
          - down_proj: (num_experts, expert_dim, hidden_size)
          - down_proj_bias: (num_experts, hidden_size)
        
        The forward pass uses INTERLEAVED gate/up layout and custom GLU:
          gate_up = x @ gate_up_proj + bias
          gate = gate_up[::2], up = gate_up[1::2]  # INTERLEAVED
          glu = gate * sigmoid(gate * alpha)
          output = (up + 1) * glu @ down_proj + down_bias
        """
        # Get expert-specific weights and biases
        gate_up_weight = experts_module.gate_up_proj[expert_idx]  # (hidden_size, expert_dim*2)
        gate_up_bias = experts_module.gate_up_proj_bias[expert_idx]  # (expert_dim*2,)
        down_weight = experts_module.down_proj[expert_idx]  # (expert_dim, hidden_size)
        down_bias = experts_module.down_proj_bias[expert_idx]  # (hidden_size,)
        
        alpha = experts_module.alpha
        limit = experts_module.limit
        
        # Compute gate_up projection: (num_tokens, expert_dim*2)
        gate_up = hidden_states @ gate_up_weight + gate_up_bias  # (num_tokens, expert_dim*2)
        
        # Split into gate and up parts using INTERLEAVED layout
        gate = gate_up[..., ::2]   # (num_tokens, expert_dim) - even indices
        up = gate_up[..., 1::2]    # (num_tokens, expert_dim) - odd indices
        
        # Apply clamp as in the actual model
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        
        # Apply GLU activation: glu = gate * sigmoid(gate * alpha)
        glu = gate * torch.sigmoid(gate * alpha)
        
        # Gated output: (up + 1) * glu
        gated_output = (up + 1) * glu
        
        # Apply down projection with bias: (num_tokens, hidden_size)
        output = gated_output @ down_weight + down_bias
        
        return output
    
    def compute_moe_output_with_experts(
        self,
        mlp_module,
        hidden_states: torch.Tensor,
        expert_subset: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute MoE layer output using only a subset of experts.
        
        IMPORTANT: This follows the correct MoE forward pass order:
        1. Compute router logits
        2. Mask non-selected experts with -inf
        3. Select top-k from LOGITS (not probabilities!)
        4. Apply softmax ONLY to those k logits
        5. Compute weighted sum of expert outputs
        
        Args:
            mlp_module: The MoE MLP module
            hidden_states: Input hidden states (num_tokens, hidden_dim)
            expert_subset: List of expert indices to use
            device: Device to run on
            
        Returns:
            Output hidden states (num_tokens, hidden_dim)
        """
        router = mlp_module.router
        experts = mlp_module.experts
        
        hidden_states = hidden_states.to(device)
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        
        # Compute router logits
        router_logits = F.linear(hidden_states, router.weight, router.bias)
        
        # Mask out non-selected experts with large negative values
        mask = torch.full((self.num_total_experts,), float('-inf'), device=device)
        for exp_idx in expert_subset:
            mask[exp_idx] = 0.0
        
        masked_logits = router_logits + mask
        
        # CRITICAL FIX: Select top-k from LOGITS first, then apply softmax
        # (not: softmax first, then top-k from probabilities)
        top_k = min(self.top_k, len(expert_subset))
        top_k_logits, top_k_indices = torch.topk(masked_logits, top_k, dim=-1)
        
        # Apply softmax ONLY to the selected top-k logits
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute expert outputs using fused parameters
        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=hidden_states.dtype)
        
        for k in range(top_k):
            expert_indices = top_k_indices[:, k]  # (num_tokens,)
            weights = top_k_weights[:, k:k+1]  # (num_tokens, 1)
            
            # Group tokens by expert
            for exp_idx in expert_subset:
                token_mask = (expert_indices == exp_idx)
                if not token_mask.any():
                    continue
                
                token_indices = token_mask.nonzero(as_tuple=True)[0]
                expert_input = hidden_states[token_indices]
                expert_weights = weights[token_indices]
                
                # Compute expert output using fused parameters
                expert_output = self.compute_single_expert_output(experts, expert_input, exp_idx)
                
                output[token_indices] += expert_weights * expert_output
        
        return output
    
    def greedy_select_experts(
        self,
        layer_idx: int,
        mlp_module,
        inputs: torch.Tensor,
        target_outputs: torch.Tensor,
    ) -> List[int]:
        """
        Greedily select experts to minimize reconstruction loss.
        Uses all cached tokens (following the paper).
        
        Args:
            layer_idx: Layer index (for logging)
            mlp_module: The MoE MLP module
            inputs: Cached input hidden states (num_tokens, hidden_dim)
            target_outputs: Cached target output states (num_tokens, hidden_dim)
            
        Returns:
            List of selected expert indices
        """
        device = next(mlp_module.parameters()).device
        
        inputs = inputs.to(device)
        target_outputs = target_outputs.to(device)
        
        num_tokens = inputs.shape[0]
        print(f"  Layer {layer_idx}: Using all {num_tokens} tokens for reconstruction")
        
        selected_experts = []
        remaining_experts = list(range(self.num_total_experts))
        
        print(f"  Layer {layer_idx}: Greedy selection ({self.num_experts_to_keep} iterations)")
        
        for iteration in range(self.num_experts_to_keep):
            best_expert = None
            best_loss = float('inf')
            
            # Try adding each remaining expert
            for candidate in remaining_experts:
                test_set = selected_experts + [candidate]
                
                # Compute output with this expert set
                with torch.no_grad():
                    output = self.compute_moe_output_with_experts(
                        mlp_module, inputs, test_set, device
                    )
                
                # Compute reconstruction loss (Frobenius norm)
                loss = torch.norm(output - target_outputs, p='fro').item()
                
                if loss < best_loss:
                    best_loss = loss
                    best_expert = candidate
            
            if best_expert is not None:
                selected_experts.append(best_expert)
                remaining_experts.remove(best_expert)
                
                if iteration < 3 or iteration == self.num_experts_to_keep - 1:
                    print(f"    Iter {iteration+1}: Added expert {best_expert}, loss={best_loss:.4f}")
        
        return selected_experts
    
    def run(
        self,
        sequences: List[torch.Tensor],
        batch_size: int = 4,
    ) -> Dict[int, List[int]]:
        """
        Run reconstruction-based expert selection with greedy addition.
        Uses all cached tokens for computing reconstruction loss.
        
        Args:
            sequences: Calibration sequences
            batch_size: Batch size for caching I/O
            
        Returns:
            Dict mapping layer_idx -> list of selected expert indices
        """
        print(f"\n[ReconstructionSelector] Starting reconstruction-based expert selection")
        print(f"[ReconstructionSelector] {len(sequences)} calibration sequences")
        
        # Step 1: Cache layer inputs and outputs
        cached_io = self.cache_layer_io(sequences, batch_size)
        
        # Step 2: Greedy selection for each layer
        selected = {}
        
        for layer_idx, name, mlp_module in tqdm(self.moe_layers, desc="Greedy selection"):
            if layer_idx not in cached_io:
                print(f"  Warning: No cached I/O for layer {layer_idx}")
                selected[layer_idx] = list(range(self.num_experts_to_keep))
                continue
            
            inputs, outputs = cached_io[layer_idx]
            
            selected[layer_idx] = self.greedy_select_experts(
                layer_idx, mlp_module, inputs, outputs
            )
        
        print(f"\n[ReconstructionSelector] Selected experts per layer (showing first 3):")
        for layer_idx in sorted(selected.keys())[:3]:
            print(f"  Layer {layer_idx}: {sorted(selected[layer_idx])}")
        
        return selected


def run_reconstruction_pruning(
    model,
    sequences: List[torch.Tensor],
    num_experts_to_keep: int = 16,
    batch_size: int = 4,
) -> Dict[int, List[int]]:
    """
    Convenience function to run reconstruction-based expert pruning.
    Uses all cached tokens for computing reconstruction loss.
    
    Args:
        model: MoE model
        sequences: Calibration sequences
        num_experts_to_keep: Number of experts to keep per layer
        batch_size: Batch size for caching
        
    Returns:
        Dict mapping layer_idx -> list of selected expert indices
    """
    selector = ReconstructionExpertSelector(model, num_experts_to_keep)
    return selector.run(sequences, batch_size)
