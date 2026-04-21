"""
Frequency-based expert pruning.

Following NAEE paper (https://arxiv.org/pdf/2402.14800):
- Count how many times each expert is selected in top-k during forward pass
- Keep the top-r most frequently activated experts per layer

This is different from "soft activation" which sums the routing weights.
We count binary activations (was expert in top-k or not).
"""

import re
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


class FrequencyExpertSelector:
    """
    Selects experts based on activation frequency over calibration data.
    """
    
    def __init__(self, model, num_experts_to_keep: int = 16):
        """
        Args:
            model: The MoE model
            num_experts_to_keep: Number of experts to keep per layer (r in NAEE)
        """
        self.model = model
        self.num_experts_to_keep = num_experts_to_keep
        
        # Find router modules and their layer indices
        self.routers = []
        for name, module in model.named_modules():
            if hasattr(module, 'router') and hasattr(module, 'experts'):
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    self.routers.append((layer_idx, name, module))
        
        self.routers.sort(key=lambda x: x[0])
        
        if self.routers:
            first_router = self.routers[0][2].router
            self.num_total_experts = first_router.weight.shape[0]
            self.top_k = getattr(first_router, 'top_k', 8)
        else:
            self.num_total_experts = 32
            self.top_k = 8
        
        print(f"[FrequencySelector] Found {len(self.routers)} MoE layers")
        print(f"[FrequencySelector] {self.num_total_experts} experts, top-{self.top_k} routing")
        print(f"[FrequencySelector] Will keep {num_experts_to_keep} experts per layer")
        
        # Expert counts: layer_idx -> Counter of expert activations
        self.expert_counts: Dict[int, Counter] = {
            layer_idx: Counter() for layer_idx, _, _ in self.routers
        }
        self.total_tokens_per_layer: Dict[int, int] = {
            layer_idx: 0 for layer_idx, _, _ in self.routers
        }
    
    def count_activations(self, sequences: List[torch.Tensor], batch_size: int = 4):
        """
        Count expert activations over calibration sequences.
        
        Args:
            sequences: List of token tensors (each shape: seq_len,)
            batch_size: Batch size for processing
        """
        self.model.eval()
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        device = next(self.model.parameters()).device
        
        # Reset counts
        for layer_idx in self.expert_counts:
            self.expert_counts[layer_idx] = Counter()
            self.total_tokens_per_layer[layer_idx] = 0
        
        # Install hooks to count activations
        hooks = []
        
        def make_hook(layer_idx, router):
            def hook(module, args):
                if len(args) == 0:
                    return
                hidden_states = args[0]
                
                # Flatten if needed
                if len(hidden_states.shape) == 3:
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    hidden_flat = hidden_states.view(-1, hidden_size)
                else:
                    hidden_flat = hidden_states
                
                with torch.no_grad():
                    router_logits = F.linear(hidden_flat, router.weight, router.bias)
                    top_k = getattr(router, 'top_k', self.top_k)
                    _, indices = torch.topk(router_logits, top_k, dim=-1)
                    
                    # Count each expert activation (binary count)
                    for idx in indices.view(-1).tolist():
                        self.expert_counts[layer_idx][idx] += 1
                    self.total_tokens_per_layer[layer_idx] += indices.shape[0]
            
            return hook
        
        # Install hooks on MLP modules (pre-hook to get input hidden states)
        for layer_idx, name, mlp_module in self.routers:
            router = mlp_module.router
            hook = mlp_module.register_forward_pre_hook(make_hook(layer_idx, router))
            hooks.append(hook)
        
        print(f"[FrequencySelector] Installed {len(hooks)} hooks")
        
        # Process sequences
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Counting activations"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(sequences))
            batch_seqs = sequences[start:end]
            
            # Stack into batch
            input_ids = torch.stack(batch_seqs).to(device)
            
            with torch.no_grad():
                # Just forward pass, we don't need output
                _ = self.model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        self.model.config.use_cache = use_cache
        torch.cuda.empty_cache()
        
        # Print statistics
        print("\n[FrequencySelector] Activation counts per layer:")
        for layer_idx in sorted(self.expert_counts.keys())[:3]:  # Show first 3 layers
            counts = self.expert_counts[layer_idx]
            total = self.total_tokens_per_layer[layer_idx]
            top_5 = counts.most_common(5)
            print(f"  Layer {layer_idx}: total={total}, top-5 experts: {top_5}")
    
    def get_selected_experts(self) -> Dict[int, List[int]]:
        """
        Get the top-r most frequently activated experts per layer.
        
        Returns:
            Dict mapping layer_idx -> list of expert indices to keep
        """
        selected = {}
        
        for layer_idx, counts in self.expert_counts.items():
            if not counts:
                # If no counts, use first r experts
                selected[layer_idx] = list(range(self.num_experts_to_keep))
                print(f"  Warning: No counts for layer {layer_idx}, using default experts")
            else:
                # Get top-r most frequent experts
                top_experts = [exp for exp, _ in counts.most_common(self.num_experts_to_keep)]
                selected[layer_idx] = top_experts
        
        return selected
    
    def run(self, sequences: List[torch.Tensor], batch_size: int = 4) -> Dict[int, List[int]]:
        """
        Run frequency-based selection.
        
        Args:
            sequences: Calibration sequences
            batch_size: Batch size
            
        Returns:
            Dict mapping layer_idx -> list of selected expert indices
        """
        print(f"\n[FrequencySelector] Starting frequency-based expert selection")
        print(f"[FrequencySelector] {len(sequences)} calibration sequences")
        
        self.count_activations(sequences, batch_size)
        selected = self.get_selected_experts()
        
        print(f"\n[FrequencySelector] Selected experts per layer (showing first 3):")
        for layer_idx in sorted(selected.keys())[:3]:
            print(f"  Layer {layer_idx}: {sorted(selected[layer_idx])}")
        
        return selected


def run_frequency_pruning(
    model,
    sequences: List[torch.Tensor],
    num_experts_to_keep: int = 16,
    batch_size: int = 4,
) -> Dict[int, List[int]]:
    """
    Convenience function to run frequency-based expert pruning.
    
    Args:
        model: MoE model
        sequences: Calibration sequences (list of token tensors)
        num_experts_to_keep: Number of experts to keep per layer
        batch_size: Batch size for forward passes
        
    Returns:
        Dict mapping layer_idx -> list of selected expert indices
    """
    selector = FrequencyExpertSelector(model, num_experts_to_keep)
    return selector.run(sequences, batch_size)
