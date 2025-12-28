# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import nullcontext

from .deepspeed import is_deepspeed_zero3_enabled
from ..models.gpt_oss.modeling_gpt_oss import (
    GptOssControllerLayerState,
    _plackett_luce_logprob,
    _runtime_flag,
    _runtime_get,
    _safe_log1m,
    _safe_logprob,
)
from ..utils import is_accelerate_available, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

if is_accelerate_available():
    from accelerate import init_empty_weights

import re
from contextlib import contextmanager


logger = logging.get_logger(__name__)

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


@contextmanager
def on_device(dev):
    if is_torch_available():
        import torch

        if isinstance(dev, torch.Tensor):
            dev = dev.device
        elif isinstance(dev, str):
            dev = torch.device(dev)
        dev_type = getattr(dev, "type", None)
        if dev_type == "cuda":
            with torch.cuda.device(dev):
                yield
                return
        if dev_type == "xpu" and hasattr(torch, "xpu"):
            with torch.xpu.device(dev):
                yield
                return
    # other: CPU
    yield


# Copied from GPT_OSS repo and vllm
def quantize_to_mxfp4(w, triton_kernels_hub):
    downcast_to_mxfp_torch = triton_kernels_hub.numerics_details.mxfp.downcast_to_mxfp_torch
    w, w_scale = downcast_to_mxfp_torch(w.to(torch.bfloat16), torch.uint8, axis=1)
    return w, w_scale


def swizzle_mxfp4(w, w_scale, triton_kernels_hub):
    """
    Changes the layout of the tensors depending on the hardware
    """
    FP4, convert_layout, wrap_torch_tensor = (
        triton_kernels_hub.tensor.FP4,
        triton_kernels_hub.tensor.convert_layout,
        triton_kernels_hub.tensor.wrap_torch_tensor,
    )
    layout = triton_kernels_hub.tensor_details.layout
    StridedLayout = triton_kernels_hub.tensor_details.layout.StridedLayout

    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale


# Copied from GPT_OSS repo
# TODO: Add absolute link when the repo is public
def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,  # TODO these values are not here by mistake ;)
) -> torch.Tensor:
    """
    Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
    pass of GPT_OSS.
    """
    import math

    # Check if blocks and scales are on CPU, and move to GPU if so
    if not blocks.is_cuda and torch.cuda.is_available():
        blocks = blocks.cuda()
        scales = scales.cuda()

    scales = scales.to(torch.int32) - 127  # TODO that's because 128=2**7

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp, sub

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    del blocks, scales, lut
    return out.transpose(1, 2).contiguous()


class Mxfp4GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.gate_up_proj_blocks = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, 16, dtype=torch.uint8),
            requires_grad=False,
        )
        self.gate_up_proj_scales = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, dtype=torch.uint8),
            requires_grad=False,
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32), requires_grad=False
        )

        self.down_proj_blocks = nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32, 16), dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_scales = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, self.intermediate_size // 32, dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, dtype=torch.float32), requires_grad=False
        )
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)
        self.gate_up_proj_precision_config = None
        self.down_proj_precision_config = None
        self.limit = getattr(config, "swiglu_limit", 7.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_data,
        gather_idx,
        scatter_idx,
        controller_state=None,
        controller_runtime=None,
    ):
        FnSpecs, FusedActivation, matmul_ogs = (
            triton_kernels_hub.matmul_ogs.FnSpecs,
            triton_kernels_hub.matmul_ogs.FusedActivation,
            triton_kernels_hub.matmul_ogs.matmul_ogs,
        )
        swiglu_fn = triton_kernels_hub.swiglu.swiglu_fn

        with on_device(hidden_states.device):
            act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (self.alpha, self.limit), 2)

            intermediate_cache1 = matmul_ogs(
                hidden_states,
                self.gate_up_proj,
                self.gate_up_proj_bias.to(torch.float32),
                routing_data,
                gather_indx=gather_idx,
                precision_config=self.gate_up_proj_precision_config,
                gammas=None,
                fused_activation=act,
            )

            intermediate_cache3 = matmul_ogs(
                intermediate_cache1,
                self.down_proj,
                self.down_proj_bias.to(torch.float32),
                routing_data,
                scatter_indx=scatter_idx,
                precision_config=self.down_proj_precision_config,
                gammas=routing_data.gate_scal,
            )
        return intermediate_cache3


# Adapted from GPT_OSS repo
# TODO: Add absolute link when the repo is public
def routing_torch_dist(
    logits,
    n_expts_act,
):
    import os

    GatherIndx, RoutingData, ScatterIndx, compute_expt_data_torch = (
        triton_kernels_hub.routing.GatherIndx,
        triton_kernels_hub.routing.RoutingData,
        triton_kernels_hub.routing.ScatterIndx,
        triton_kernels_hub.routing.compute_expt_data_torch,
    )

    with on_device(logits.device):
        world_size = torch.distributed.get_world_size()
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        replace_value = -1

        n_tokens = logits.shape[0]
        n_expts_tot = logits.shape[1]

        n_local_experts = n_expts_tot // world_size
        local_expert_start = rank * n_local_experts
        local_expert_end = (rank + 1) * n_local_experts

        n_gates_pad = n_tokens * n_expts_act

        def topk(vals, k):
            tk_indx = torch.argsort(-vals, dim=1, stable=True)[:, :k]
            tk_indx = tk_indx.long()
            tk_val = torch.take_along_dim(vals, tk_indx, dim=1)
            return tk_val, tk_indx.int()

        expt_scal, expt_indx = topk(logits, n_expts_act)
        expt_scal = torch.softmax(expt_scal, dim=-1)
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)

        # Flatten and mask for local experts
        expt_scal = expt_scal.reshape(-1)

        hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1)[local_expert_start:local_expert_end]

        expt_indx = expt_indx.view(-1).to(torch.int32)

        # we use a large value to replace the indices that are not in the local expert range
        var = 1000
        expt_indx = torch.where(expt_indx < local_expert_start, var, expt_indx)
        topk_indx = torch.argsort(expt_indx, stable=True).to(torch.int32)
        gate_indx = torch.argsort(topk_indx).to(torch.int32)
        expt_indx = torch.where(expt_indx < local_expert_end, expt_indx, replace_value)
        expt_indx = torch.where(local_expert_start <= expt_indx, expt_indx, replace_value)

        gate_indx = torch.where(expt_indx == replace_value, replace_value, gate_indx)
        gate_scal = expt_scal[topk_indx]

        topk_indx = torch.where(gate_indx[topk_indx] == replace_value, replace_value, topk_indx)

        # # Routing metadata for local expert computation
        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())

        expt_data = compute_expt_data_torch(hist, n_local_experts, n_gates_pad)

        hit_experts = n_expts_act
    return RoutingData(gate_scal, hist, n_local_experts, hit_experts, expt_data), gather_indx, scatter_indx


def mlp_forward(
    self,
    hidden_states,
    controller_state=None,
    controller_runtime=None,
    **kwargs,
):
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and hasattr(self, "_is_hooked"):
        routing = routing_torch_dist
    else:
        routing = triton_kernels_hub.routing.routing

    # Normalize hidden state rank. Some inference paths may squeeze the batch/seq dims when both equal 1,
    # which later breaks the router linear projection expecting a 2D matrix.
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)
    elif hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)

    batch_size, seq_len, _ = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, self.router.hidden_dim)

    router_weight = self.router.weight
    router_bias = self.router.bias

    def _reshape_router_weight(weight):
        if weight.dim() == 1 and weight.numel() % self.router.hidden_dim == 0:
            return weight.reshape(-1, self.router.hidden_dim)
        return weight

    def _enter_zero3_ctx(param):
        if not (is_deepspeed_zero3_enabled() and param is not None):
            return None
        import deepspeed

        ctx = deepspeed.zero.GatheredParameters([param], modifier_rank=0)
        ctx.__enter__()
        return ctx

    def _exit_zero3_ctx(ctx, param):
        if ctx is not None and param is not None:
            active = getattr(param, "ds_active_sub_modules", None)
            if active is not None:
                active.clear()
            ctx.__exit__(None, None, None)

    weight_ctx = _enter_zero3_ctx(router_weight)
    bias_ctx = _enter_zero3_ctx(router_bias)

    try:
        router_weight = _reshape_router_weight(self.router.weight)
        router_bias = self.router.bias
        router_logits = nn.functional.linear(hidden_states, router_weight, router_bias)
    finally:
        _exit_zero3_ctx(bias_ctx, router_bias)
        _exit_zero3_ctx(weight_ctx, router_weight)

    original_hidden_dtype = hidden_states.dtype

    def _dispatch_experts(input_states, routing_data, gather_idx, scatter_idx):
        expert_states = input_states
        if expert_states.dtype == torch.float16:
            expert_states = expert_states.to(torch.bfloat16)
        routed = self.experts(expert_states, routing_data, gather_idx, scatter_idx)
        if routed.dtype != original_hidden_dtype:
            routed = routed.to(original_hidden_dtype)
        return routed

    if not getattr(self, "controller_enabled", False):
        with on_device(router_logits.device):
            routing_data, gather_idx, scatter_idx = routing(router_logits, self.router.top_k)
            routed_out = _dispatch_experts(hidden_states, routing_data, gather_idx, scatter_idx)
            routed_out = routed_out.reshape(batch_size, seq_len, self.router.hidden_dim)
            return routed_out, router_logits, controller_state

    router_logits_tokens = router_logits.view(batch_size, seq_len, -1)
    # Keep per-token hidden states for controller (needed when input_type="hidden_states")
    hidden_states_tokens = hidden_states.view(batch_size, seq_len, -1)
    allowed_mask = controller_state.allowed_mask if controller_state is not None else None
    hidden_state = controller_state.hidden_state if controller_state is not None else None
    # For activation controller, track current_expert_indices
    current_expert_indices = controller_state.current_expert_indices if (controller_state is not None and hasattr(controller_state, 'current_expert_indices')) else None
    use_activation_controller = getattr(self, "controller_type", "rnn") == "activation"
    
    switch_trace = []
    switch_logprob_trace = []
    selection_logprob_trace = []
    selected_indices_trace = []
    controller_input_trace = []
    q_u_old_trace = []

    layer_idx = getattr(self, "layer_idx", None)
    record_actions = _runtime_get(controller_runtime, "record_actions", None)
    replay_actions = _runtime_get(controller_runtime, "replay_actions", None)
    layer_replay = None
    if replay_actions is not None and layer_idx is not None:
        layer_replay = replay_actions.get(layer_idx)

    mask_value = torch.finfo(router_logits.dtype).min

    def _ensure_replay_capacity(layer_dict, required_len):
        if layer_dict is None:
            return None
        current_len = layer_dict["switches"].shape[1]
        if current_len >= required_len:
            return layer_dict
        pad_len = required_len - current_len

        def _pad(tensor, fill_value):
            pad_shape = list(tensor.shape)
            pad_shape[1] = pad_len
            padding = tensor.new_full(pad_shape, fill_value)
            return torch.cat([tensor, padding], dim=1)

        layer_dict = {
            "switches": _pad(layer_dict["switches"], False),
            "switch_logprobs": _pad(layer_dict["switch_logprobs"], 0.0),
            "selection_logprobs": _pad(layer_dict["selection_logprobs"], 0.0),
            "selected_indices": _pad(layer_dict["selected_indices"], -1),
        }
        return layer_dict

    def _controller_cached_logprob(layer_cache):
        # Canonical RNN controller: need to run sequentially with hidden state
        controller_inputs = layer_cache.get("controller_inputs")
        if controller_inputs is None:
            return None
        switches = layer_cache["switches"]
        selected_indices = layer_cache["selected_indices"]
        batch_size, seq_len = switches.shape
        device = router_logits_tokens.device
        
        # Sequential processing for canonical RNN
        inputs = controller_inputs.to(device=device)  # [batch, seq, 2*num_experts]
        hidden_dim = self.controller.hidden_dim
        h = self.controller.init_hidden(batch_size, device, inputs.dtype)
        
        all_switch_logits = []
        all_candidate_logits = []
        
        for t in range(seq_len):
            x_t = inputs[:, t, :]  # [batch, 2*num_experts]
            h, switch_logits_t, candidate_logits_t, _ = self.controller(x_t, h)
            all_switch_logits.append(switch_logits_t)
            all_candidate_logits.append(candidate_logits_t)
        
        switch_logits = torch.stack(all_switch_logits, dim=1)  # [batch, seq]
        candidate_logits = torch.stack(all_candidate_logits, dim=1)  # [batch, seq, num_experts]
        
        switch_probs = torch.sigmoid(switch_logits)
        switch_decisions = switches.to(device=device, dtype=torch.bool)
        switch_logprob = torch.where(
            switch_decisions,
            _safe_logprob(switch_probs),
            _safe_log1m(switch_probs),
        )
        # Compute expert selection log prob for ALL timesteps
        # We now record selected_indices for ALL timesteps (even when switch=False),
        # so we can simply compute the log_prob for all of them.
        # This balances log_prob magnitudes between switch and no-switch actions.
        flat_candidates = candidate_logits.view(batch_size * seq_len, -1)
        flat_selected = selected_indices.to(device=device, dtype=torch.long).view(batch_size * seq_len, -1)
        selection_logprob = _plackett_luce_logprob(flat_candidates, flat_selected)
        selection_logprob = selection_logprob.view(batch_size, seq_len)
        return switch_logprob + selection_logprob

    logprob_cache = _runtime_get(controller_runtime, "logprob_cache", None)
    record_logprobs = _runtime_get(controller_runtime, "record_logprobs", None)

    # Traces for canonical RNN controller architecture
    value_trace = []
    controller_input_trace = []  # Now stores x_t = [router_softmax, expert_mask]
    router_logits_trace = []

    # Determine if this is the very first token (no prior controller state)
    # Only the first token (t=0) uses router top-k to initialize
    # All subsequent tokens (t>0, whether prefill or generation) use normal controller operation
    is_first_forward = controller_state is None
    
    for token_idx in range(seq_len):
        token_logits = router_logits_tokens[:, token_idx, :]
        raw_token_logits = token_logits
        # Convert router logits to softmax for controller input (consistent with training)
        # Clamp to avoid numerical issues (same as training)
        router_softmax = torch.softmax(raw_token_logits.float().clamp(-50, 50), dim=-1).to(raw_token_logits.dtype)
        is_first_token = token_idx == 0 and is_first_forward
        
        if is_first_token:
            # t=0: Initialize with router top-k, no switch decision
            # Initialize hidden state and allowed_mask
            allowed_mask = torch.zeros_like(raw_token_logits, dtype=torch.bool)
            top_allowed = torch.topk(raw_token_logits, self.controller_allowed_experts, dim=-1).indices
            allowed_mask.scatter_(1, top_allowed, True)
            switch_decision = torch.zeros(raw_token_logits.shape[0], dtype=torch.bool, device=raw_token_logits.device)
            # Use float32 for log probabilities (controller outputs are float32 for numerical stability)
            switch_logprob = torch.zeros(raw_token_logits.shape[0], dtype=torch.float32, device=raw_token_logits.device)
            selection_logprob = torch.zeros(raw_token_logits.shape[0], dtype=torch.float32, device=raw_token_logits.device)
            selected_indices = top_allowed  # Use actual top-k indices
            value = torch.zeros(raw_token_logits.shape[0], dtype=torch.float32, device=raw_token_logits.device)
            q_u_old = torch.zeros(raw_token_logits.shape[0], dtype=torch.float32, device=raw_token_logits.device)
            controller_input = torch.cat([router_softmax, allowed_mask.to(dtype=raw_token_logits.dtype)], dim=-1)
            
            if use_activation_controller:
                # Activation controller: initialize current_expert_indices with router top-k
                current_expert_indices = top_allowed
            else:
                # RNN controller: initialize hidden state and run forward
                hidden_state = self.controller.init_hidden(raw_token_logits.shape[0], raw_token_logits.device, raw_token_logits.dtype)
                # Controller input for recording: depends on input_type
                if self.controller.input_type == "hidden_states":
                    token_hidden = hidden_states_tokens[:, token_idx, :]
                    controller_input = torch.cat([token_hidden, allowed_mask.to(dtype=token_hidden.dtype)], dim=-1)
                else:
                    controller_input = torch.cat([router_softmax, allowed_mask.to(dtype=raw_token_logits.dtype)], dim=-1)
                
                # Run controller to initialize hidden state for t=1
                x_t = controller_input.to(dtype=self.controller.gru_cell.weight_ih.dtype)
                hidden_state, _, _, _ = self.controller(x_t, hidden_state)
        else:
            # t>0: Normal controller operation (both prefill and generation)
            # Controller can decide to switch experts based on learned policy
            replay_payload = None
            if layer_replay is not None:
                layer_replay = _ensure_replay_capacity(layer_replay, token_idx + 1)
                replay_payload = {
                    "switches": layer_replay["switches"][:, token_idx],
                    "selected_indices": layer_replay["selected_indices"][:, token_idx],
                }
                if "switch_logprobs" in layer_replay:
                    replay_payload["switch_logprobs"] = layer_replay["switch_logprobs"][:, token_idx]
                if "selection_logprobs" in layer_replay:
                    replay_payload["selection_logprobs"] = layer_replay["selection_logprobs"][:, token_idx]
            
            if use_activation_controller:
                # Activation controller: use LLM hidden states + current expert indices
                token_hidden = hidden_states_tokens[:, token_idx, :]
                (
                    switch_decision,
                    current_expert_indices,  # updated if switch=1
                    switch_logprob,
                    selection_logprob,
                    selected_indices,
                    value,
                    q_u_old,
                ) = self._apply_activation_controller_step(
                    raw_token_logits,    # Raw router logits for residual connection
                    current_expert_indices,  # Current option (expert set)
                    token_hidden,    # LLM hidden states
                    controller_runtime,
                    replay_payload=replay_payload,
                )
                # Update allowed_mask based on current_expert_indices
                allowed_mask = torch.zeros_like(raw_token_logits, dtype=torch.bool)
                allowed_mask.scatter_(1, current_expert_indices, True)
                controller_input = torch.cat([router_softmax, allowed_mask.to(dtype=raw_token_logits.dtype)], dim=-1)
            else:
                # RNN controller
                (
                    switch_decision,
                    allowed_mask,
                    hidden_state,
                    switch_logprob,
                    selection_logprob,
                    selected_indices,
                    value,
                    controller_input,
                ) = self._apply_controller_step(
                    router_softmax,
                    raw_token_logits,
                    allowed_mask,
                    hidden_state,
                    controller_runtime,
                    replay_payload=replay_payload,
                    mlp_hidden_states=hidden_states_tokens[:, token_idx, :],
                )
                q_u_old = torch.zeros(raw_token_logits.shape[0], dtype=torch.float32, device=raw_token_logits.device)

        # Record traces BEFORE masking (use .clone() since detach() shares storage)
        # CRITICAL: raw_token_logits is a view into router_logits_tokens, so we must clone
        # before the in-place masking overwrites the storage
        router_logits_trace.append(raw_token_logits.detach().clone())
        switch_trace.append(switch_decision)
        switch_logprob_trace.append(switch_logprob)
        selection_logprob_trace.append(selection_logprob)
        selected_indices_trace.append(selected_indices)
        controller_input_trace.append(controller_input.detach().clone())
        value_trace.append(value)
        q_u_old_trace.append(q_u_old)
        
        # Apply mask to router logits (this overwrites router_logits_tokens storage)
        masked_logits = raw_token_logits.masked_fill(~allowed_mask, mask_value)
        router_logits_tokens[:, token_idx, :] = masked_logits

    masked_router_logits = router_logits_tokens.reshape(batch_size * seq_len, -1)
    with on_device(masked_router_logits.device):
        routing_data, gather_idx, scatter_idx = routing(masked_router_logits, self.router.top_k)
    routed_out = _dispatch_experts(hidden_states, routing_data, gather_idx, scatter_idx)
    routed_out = routed_out.reshape(batch_size, seq_len, self.router.hidden_dim)

    # Build next state based on controller type
    if use_activation_controller:
        next_state = GptOssControllerLayerState(
            allowed_mask=allowed_mask,
            hidden_state=None,
            current_expert_indices=current_expert_indices,
        )
    else:
        next_state = GptOssControllerLayerState(
            allowed_mask=allowed_mask,
            hidden_state=hidden_state,
        )

    if record_actions is not None and layer_idx is not None:
        new_record = {
            "switches": torch.stack(switch_trace, dim=1),
            "switch_logprobs": torch.stack(switch_logprob_trace, dim=1),
            "selection_logprobs": torch.stack(selection_logprob_trace, dim=1),
            "selected_indices": torch.stack(selected_indices_trace, dim=1),
            "values": torch.stack(value_trace, dim=1),
            "q_u_old_values": torch.stack(q_u_old_trace, dim=1),  # For activation controller
            "router_logits": torch.stack(router_logits_trace, dim=1),
            "controller_inputs": torch.stack(controller_input_trace, dim=1),
            # Use hidden_states_tokens (shape [batch, seq, hidden]) not hidden_states (flattened)
            "llm_hidden_states": hidden_states_tokens.detach().clone(),
        }
        if layer_idx in record_actions:
            prev_record = record_actions[layer_idx]
            new_record = {
                key: torch.cat([prev_record[key], val], dim=1) for key, val in new_record.items()
            }
        record_actions[layer_idx] = {key: val.detach() for key, val in new_record.items()}

    # Record actual controller log-probabilities computed in this forward pass so gradients flow correctly.
    if record_logprobs is not None and layer_idx is not None:
        record_logprobs[layer_idx] = {
            "switch_logprobs": torch.stack(switch_logprob_trace, dim=1),
            "selection_logprobs": torch.stack(selection_logprob_trace, dim=1),
        }

    if _runtime_flag(controller_runtime, "record_stats", False):
        stats_collector = _runtime_get(controller_runtime, "stats", None)
        if stats_collector is not None:
            stats_collector.append(
                {
                    "switches": torch.stack(switch_trace, dim=1),
                    "allowed_mask": allowed_mask,
                    "hidden_state": hidden_state,  # Canonical RNN: store hidden state
                }
            )

    return routed_out, masked_router_logits, next_state


def should_convert_module(current_key_name, patterns):
    current_key_name_str = ".".join(current_key_name)
    if not any(
        re.match(f"{key}\\.", current_key_name_str) or re.match(f"{key}", current_key_name_str) for key in patterns
    ):
        return True
    return False


def dequantize(module, param_name, param_value, target_device, dq_param_name, **kwargs):
    from ..integrations.tensor_parallel import shard_and_distribute_module

    model = kwargs.get("model")
    empty_param = kwargs.get("empty_param")
    casting_dtype = kwargs.get("casting_dtype")
    to_contiguous = kwargs.get("to_contiguous")
    rank = kwargs.get("rank")
    device_mesh = kwargs.get("device_mesh")

    for proj in ["gate_up_proj", "down_proj"]:
        if proj in param_name:
            if device_mesh is not None:
                param_value = shard_and_distribute_module(
                    model,
                    param_value,
                    empty_param,
                    dq_param_name,
                    casting_dtype,
                    to_contiguous,
                    rank,
                    device_mesh,
                )
            blocks_attr = f"{proj}_blocks"
            scales_attr = f"{proj}_scales"
            setattr(module, param_name.rsplit(".", 1)[1], param_value)
            if hasattr(module, blocks_attr) and hasattr(module, scales_attr):
                dequantized = convert_moe_packed_tensors(getattr(module, blocks_attr), getattr(module, scales_attr))
                if target_device == "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                setattr(module, proj, torch.nn.Parameter(dequantized.to(target_device)))
                delattr(module, blocks_attr)
                delattr(module, scales_attr)


def load_and_swizzle_mxfp4(module, param_name, param_value, target_device, triton_kernels_hub, **kwargs):
    """
    This transforms the weights obtained using `convert_gpt_oss.py` to load them into `Mxfp4GptOssExperts`.
    """
    PrecisionConfig, FlexCtx, InFlexData = (
        triton_kernels_hub.matmul_ogs.PrecisionConfig,
        triton_kernels_hub.matmul_ogs.FlexCtx,
        triton_kernels_hub.matmul_ogs.InFlexData,
    )
    from ..integrations.tensor_parallel import shard_and_distribute_module

    model = kwargs.get("model")
    empty_param = kwargs.get("empty_param")
    casting_dtype = kwargs.get("casting_dtype")
    to_contiguous = kwargs.get("to_contiguous")
    rank = kwargs.get("rank")
    device_mesh = kwargs.get("device_mesh")
    if "blocks" in param_name:
        proj = param_name.split(".")[-1].split("_blocks")[0]
        part = "blocks"
    elif "scales" in param_name:
        proj = param_name.split(".")[-1].split("_scales")[0]
        part = "scales"
    else:
        proj = None
        part = None
    if not hasattr(module, "_mxfp4_loaded_parts"):
        module._mxfp4_loaded_parts = {}
    if proj is not None:
        module._mxfp4_loaded_parts.setdefault(proj, set()).add(part)
    if device_mesh is not None:
        shard_and_distribute_module(
            model, param_value, empty_param, param_name, casting_dtype, to_contiguous, rank, device_mesh
        )
    else:
        setattr(module, param_name.rsplit(".", 1)[1], torch.nn.Parameter(param_value, requires_grad=False))
    blocks_attr = f"{proj}_blocks"
    scales_attr = f"{proj}_scales"
    blocks = getattr(module, blocks_attr)  # at this point values were loaded from ckpt
    scales = getattr(module, scales_attr)
    loaded_parts = module._mxfp4_loaded_parts.get(proj, set())
    if loaded_parts != {"blocks", "scales"}:
        return
    if blocks.device.type == "meta" or scales.device.type == "meta":
        return
    # Check if both blocks and scales both not on meta device
    local_experts = blocks.size(0)
    if proj == "gate_up_proj":
        blocks = blocks.reshape(local_experts, module.intermediate_size * 2, -1)
    else:
        blocks = blocks.reshape(local_experts, -1, module.intermediate_size // 2)
    if getattr(target_device, "type", target_device) == "cpu":
        target_device = "cuda"
    blocks = blocks.to(target_device).contiguous()
    scales = scales.to(target_device).contiguous()
    with on_device(target_device):
        triton_weight_tensor, weight_scale = swizzle_mxfp4(
            blocks.transpose(-2, -1), scales.transpose(-2, -1), triton_kernels_hub
        )

    # need to overwrite the shapes for the kernels
    if proj == "gate_up_proj":
        triton_weight_tensor.shape = torch.Size([local_experts, module.hidden_size, module.intermediate_size * 2])
    else:
        triton_weight_tensor.shape = torch.Size([local_experts, module.intermediate_size, module.hidden_size])

    # triton_weight_tensor is what needs to be passed in oai kernels. It stores the data, the shapes and any more objects. It is like a subtensor
    setattr(module, proj, triton_weight_tensor)
    setattr(
        module,
        f"{proj}_precision_config",
        PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
    )

    # Remove the quantized parameter containers entirely to avoid ZeRO trying to shard zero-sized tensors.
    if hasattr(module, scales_attr):
        delattr(module, scales_attr)
    if hasattr(module, blocks_attr):
        delattr(module, blocks_attr)

    # Replace with tiny placeholder parameters (numel>0) so ZeRO won't crash while still keeping attrs accessible.
    setattr(module, scales_attr, torch.nn.Parameter(torch.zeros(1, dtype=torch.uint8), requires_grad=False))
    setattr(module, blocks_attr, torch.nn.Parameter(torch.zeros(1, dtype=torch.uint8), requires_grad=False))
    del blocks
    module._mxfp4_loaded_parts.pop(proj, None)


def _replace_with_mxfp4_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    config=None,
):
    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)
        if not should_convert_module(current_key_name, modules_to_not_convert):
            current_key_name.pop(-1)
            continue
        if module.__class__.__name__ == "GptOssExperts" and not quantization_config.dequantize:
            with init_empty_weights():
                model._modules[name] = Mxfp4GptOssExperts(config)
                has_been_replaced = True
        if module.__class__.__name__ == "GptOssMLP" and not quantization_config.dequantize:
            from types import MethodType

            module.forward = MethodType(mlp_forward, module)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_mxfp4_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                config=config,
            )
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_mxfp4_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    config=None,
):
    if quantization_config.dequantize:
        return model
    else:
        from kernels import get_kernel

        global triton_kernels_hub
        triton_kernels_hub = get_kernel("kernels-community/triton_kernels")

    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_mxfp4_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        quantization_config,
        config=config,
    )
    if not has_been_replaced:
        logger.warning(
            "You are loading your model using mixed-precision FP4 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
