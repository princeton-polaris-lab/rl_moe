"""
Canonical Wanda weight pruning for MoE models.

Faithfully follows the official implementation at https://github.com/locuslab/wanda
(Sun et al., "A Simple and Effective Pruning Approach for Large Language Models", ICLR 2024)

Key properties matching the official implementation:
1. Layer-by-layer sequential processing: prune one transformer layer at a time,
   then re-run calibration through the pruned layer to update activations for next layer.
2. Per-sublayer activation collection: each linear layer (and each MoE expert projection)
   collects its own input activation norms via hooks.
3. ALL linear layers pruned: attention projections + expert gate_up_proj + expert down_proj.
4. Wanda metric: S_ij = |W_ij| * sqrt(scaler_row_j), per-output pruning comparison.

Adaptation for GptOss MoE architecture:
- Attention q/k/v/o_proj are standard nn.Linear → handled like official Wanda.
- Expert gate_up_proj and down_proj are fused Parameters (num_experts, in, out),
  not nn.Linear. We collect per-expert activation norms by hooking into the experts
  forward pass and tracking routing decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import re


class WrappedLinear:
    """Wraps an nn.Linear to collect input activation statistics (matching official WrappedGPT)."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros(self.columns, device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t()  # (in_features, num_tokens)

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


class ExpertActivationCollector:
    """Collects per-expert input activation norms for gate_up_proj and down_proj."""

    def __init__(self, experts_module, num_experts: int, hidden_size: int, expert_dim: int, device):
        self.experts_module = experts_module
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.expert_dim = expert_dim
        self.device = device

        self.gate_up_scaler = [
            torch.zeros(hidden_size, device=device) for _ in range(num_experts)
        ]
        self.down_scaler = [
            torch.zeros(expert_dim, device=device) for _ in range(num_experts)
        ]
        self.nsamples = [0] * num_experts

    def collect_from_forward(self, hidden_states, router_indices, routing_weights):
        """
        Manually compute per-expert activations following GptOssExperts.forward logic.

        Args:
            hidden_states: (num_tokens, hidden_size) — input to experts
            router_indices: (num_tokens, top_k) — which experts each token goes to
            routing_weights: (num_tokens, num_experts) — routing weights
        """
        experts = self.experts_module
        num_experts = routing_weights.shape[1]

        expert_mask = F.one_hot(router_indices.long(), num_classes=num_experts + 1)
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            _, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states[token_idx]  # (n_routed, hidden_size)
            n_tokens = current_state.shape[0]

            inp_t = current_state.t().float()  # (hidden_size, n_routed)
            self.gate_up_scaler[expert_idx] *= self.nsamples[expert_idx] / (self.nsamples[expert_idx] + 1)
            self.nsamples[expert_idx] += 1
            self.gate_up_scaler[expert_idx] += torch.norm(inp_t, p=2, dim=1) ** 2 / self.nsamples[expert_idx]

            gate_up = current_state @ experts.gate_up_proj[expert_idx] + experts.gate_up_proj_bias[expert_idx]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=experts.limit)
            up = up.clamp(min=-experts.limit, max=experts.limit)
            glu = gate * torch.sigmoid(gate * experts.alpha)
            gated_output = (up + 1) * glu  # (n_routed, expert_dim) — input to down_proj

            down_inp_t = gated_output.t().float()  # (expert_dim, n_routed)
            self.down_scaler[expert_idx] *= (self.nsamples[expert_idx] - 1) / self.nsamples[expert_idx]
            self.down_scaler[expert_idx] += torch.norm(down_inp_t, p=2, dim=1) ** 2 / self.nsamples[expert_idx]


def _find_linear_layers(module):
    """Find all nn.Linear sublayers (excluding router)."""
    layers = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and 'router' not in name:
            layers[name] = child
    return layers


def _prune_weight(W, scaler_row, sparsity_type, sparsity_ratio, structured_n, structured_m):
    """
    Apply Wanda pruning to a weight matrix.
    Follows official implementation exactly.

    Args:
        W: weight tensor (out_features, in_features) for nn.Linear
        scaler_row: accumulated activation norms (in_features,)
        sparsity_type: "unstructured" or "structured"
        sparsity_ratio: fraction to prune (for unstructured)
        structured_n: N in N:M (for structured)
        structured_m: M in N:M (for structured)
    """
    W_metric = torch.abs(W) * torch.sqrt(scaler_row.reshape(1, -1))

    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

    if sparsity_type == "structured" and structured_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % structured_m == 0:
                tmp = W_metric[:, ii:(ii + structured_m)].float()
                W_mask.scatter_(
                    1,
                    ii + torch.topk(tmp, structured_n, dim=1, largest=False)[1],
                    True,
                )
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
        W_mask.scatter_(1, indices, True)

    W[W_mask] = 0


def _prune_expert_weight(W_param, scaler, sparsity_type, sparsity_ratio, structured_n, structured_m, transposed=False):
    """
    Prune a single expert's weight parameter.

    For GptOssExperts, weights are stored as (in_features, out_features) — transposed
    relative to nn.Linear (out_features, in_features). The operation is x @ W.

    We transpose to standard form, apply Wanda, then transpose back.
    """
    if transposed:
        W = W_param.data.T.clone()  # (out_features, in_features)
    else:
        W = W_param.data.clone()

    _prune_weight(W, scaler, sparsity_type, sparsity_ratio, structured_n, structured_m)

    if transposed:
        W_param.data.copy_(W.T)
    else:
        W_param.data.copy_(W)


def _prepare_calibration_input(model, sequences, batch_size, device):
    """
    Prepare calibration inputs by running sequences through the embedding layer.
    Returns the hidden states that would be input to the first transformer layer.
    """
    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype

    all_inps = []
    all_attention_masks = []
    all_position_ids = []

    cache = {'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            all_inps.append(inp.detach().cpu())
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        input_ids = torch.stack(batch_seqs).to(device)
        try:
            model(input_ids)
        except ValueError:
            pass

    layers[0] = layers[0].module

    inps = torch.cat(all_inps, dim=0)  # (nsamples, seq_len, hidden)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    # Slice to batch=1: the layer loop processes one sample at a time, but
    # the cached tensors have the batch size of the last Catcher batch.
    # Since all samples share the same seq_len, mask/pos are identical across the batch.
    if attention_mask is not None and attention_mask.shape[0] > 1:
        attention_mask = attention_mask[:1]
    if position_ids is not None and position_ids.shape[0] > 1:
        position_ids = position_ids[:1]
    if position_embeddings is not None:
        position_embeddings = tuple(
            pe[:1] if pe.dim() >= 1 and pe.shape[0] > 1 else pe
            for pe in position_embeddings
        )

    return inps, attention_mask, position_ids, position_embeddings


def run_wanda_pruning(
    model: nn.Module,
    sequences: List[torch.Tensor],
    sparsity_type: str = "structured",
    sparsity_ratio: float = 0.5,
    structured_n: int = 2,
    structured_m: int = 4,
    batch_size: int = 4,
) -> nn.Module:
    """
    Apply Wanda pruning layer-by-layer, following the official implementation.
    Prunes all linear layers including per-expert gate_up_proj and down_proj.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

    device = next(model.parameters()).device

    # In our convention, structured_n = number to KEEP per group of M.
    # Official Wanda uses prune_n = number to PRUNE (zero out) per group of M.
    # Convert: prune_n = M - keep_n
    prune_n = (structured_m - structured_n) if sparsity_type == "structured" else 0
    prune_m = structured_m if sparsity_type == "structured" else 0

    if sparsity_type == "structured":
        actual_sparsity = 1.0 - (structured_n / structured_m)
        print(f"[WANDA] Structured {structured_n}:{structured_m} "
              f"(keep {structured_n} of every {structured_m}, {actual_sparsity * 100:.1f}% pruned)")
    else:
        print(f"[WANDA] Unstructured, sparsity_ratio={sparsity_ratio}")

    print(f"[WANDA] Preparing calibration inputs ({len(sequences)} sequences)...")
    inps, attention_mask, position_ids, position_embeddings = _prepare_calibration_input(
        model, sequences, batch_size, device
    )
    nsamples = inps.shape[0]
    outs = torch.zeros_like(inps)
    print(f"[WANDA] Calibration input shape: {inps.shape}")

    layers = model.model.layers
    total_pruned_params = 0
    total_params = 0

    for layer_idx in tqdm(range(len(layers)), desc="[WANDA] Pruning layers"):
        layer = layers[layer_idx]
        if hasattr(layer, '_hf_hook') and hasattr(layer._hf_hook, 'execution_device'):
            layer_device = layer._hf_hook.execution_device
        else:
            layer_device = next(layer.parameters()).device

        inps = inps.to(layer_device)
        outs = outs.to(layer_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(layer_device)
        if position_ids is not None:
            position_ids = position_ids.to(layer_device)
        if position_embeddings is not None:
            position_embeddings = tuple(pe.to(layer_device) for pe in position_embeddings)

        linear_sublayers = _find_linear_layers(layer)
        wrapped = {name: WrappedLinear(mod) for name, mod in linear_sublayers.items()}

        mlp = layer.mlp
        experts = mlp.experts
        num_experts = experts.num_experts
        hidden_size = experts.hidden_size
        expert_dim = experts.expert_dim
        expert_collector = ExpertActivationCollector(
            experts, num_experts, hidden_size, expert_dim, layer_device
        )

        handles = []
        for name in wrapped:
            def make_hook(wname):
                def hook_fn(_, inp, out):
                    wrapped[wname].add_batch(inp[0].data, out.data)
                return hook_fn
            handles.append(linear_sublayers[name].register_forward_hook(make_hook(name)))

        original_experts_forward = experts.forward

        def patched_experts_forward(hidden_states, router_indices=None, routing_weights=None):
            with torch.no_grad():
                hs_flat = hidden_states.reshape(-1, hidden_size) if hidden_states.dim() == 3 else hidden_states
                expert_collector.collect_from_forward(hs_flat, router_indices, routing_weights)
            return original_experts_forward(hidden_states, router_indices=router_indices, routing_weights=routing_weights)

        experts.forward = patched_experts_forward

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]

        for h in handles:
            h.remove()
        experts.forward = original_experts_forward

        for name in wrapped:
            mod = linear_sublayers[name]
            W = mod.weight.data
            n_before = (W == 0).sum().item()
            _prune_weight(W, wrapped[name].scaler_row, sparsity_type, sparsity_ratio, prune_n, prune_m)
            n_after = (W == 0).sum().item()
            total_params += W.numel()
            total_pruned_params += n_after
            if layer_idx < 2:
                print(f"  Layer {layer_idx} {name}: {W.shape}, pruned {n_after - n_before} weights")

        for exp_idx in range(num_experts):
            gu_scaler = expert_collector.gate_up_scaler[exp_idx]
            if gu_scaler.sum() > 0:
                W_gu = experts.gate_up_proj[exp_idx]  # (hidden_size, 2*expert_dim) — transposed
                n_before = (W_gu == 0).sum().item()
                _prune_expert_weight(
                    W_gu, gu_scaler, sparsity_type, sparsity_ratio, prune_n, prune_m,
                    transposed=True,
                )
                experts.gate_up_proj.data[exp_idx] = W_gu.data
                n_after = (experts.gate_up_proj[exp_idx] == 0).sum().item()
                total_params += W_gu.numel()
                total_pruned_params += n_after

            dp_scaler = expert_collector.down_scaler[exp_idx]
            if dp_scaler.sum() > 0:
                W_dp = experts.down_proj[exp_idx]  # (expert_dim, hidden_size) — transposed
                n_before = (W_dp == 0).sum().item()
                _prune_expert_weight(
                    W_dp, dp_scaler, sparsity_type, sparsity_ratio, prune_n, prune_m,
                    transposed=True,
                )
                experts.down_proj.data[exp_idx] = W_dp.data
                n_after = (experts.down_proj[exp_idx] == 0).sum().item()
                total_params += W_dp.numel()
                total_pruned_params += n_after

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
        inps, outs = outs, inps

    actual_sparsity = total_pruned_params / total_params if total_params > 0 else 0
    print(f"\n[WANDA] Pruning complete:")
    print(f"  - Total params in pruned layers: {total_params:,}")
    print(f"  - Pruned params (zeros): {total_pruned_params:,}")
    print(f"  - Actual sparsity: {actual_sparsity * 100:.1f}%")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return model
