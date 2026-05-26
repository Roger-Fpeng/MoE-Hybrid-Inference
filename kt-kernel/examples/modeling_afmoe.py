# coding=utf-8
# Copyright 2024 Arcee AI. All rights reserved.
# Adapted for kt-kernel CPU/GPU hybrid MoE inference.
# SPDX-License-Identifier: Apache-2.0

"""
Afmoe (Trinity-Nano-Base) model adapted for kt-kernel MoE acceleration.

Key changes vs upstream:
- AfmoeMoE.forward() uses KTMoEWrapper for CPU-offloaded expert inference
- Shared experts remain on GPU
- Router outputs (topk_ids, topk_weights) fed directly to KTMoEWrapper
"""

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache, DynamicCache

try:
    from .configuration_afmoe import AfmoeConfig
except ImportError:
    from configuration_afmoe import AfmoeConfig

# kt-kernel imports
from kt_kernel import KTMoEWrapper


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class AfmoeRotaryEmbedding(nn.Module):
    def __init__(self, config: AfmoeConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float().to(x.device)
                @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Norms & MLP
# ---------------------------------------------------------------------------

class AfmoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class AfmoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class AfmoeAttention(nn.Module):
    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_local_attention else None

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        gate_states = self.gate_proj(hidden_states)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.is_local_attention:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        output = output.view(*input_shape, -1).contiguous()
        output = output * F.sigmoid(gate_states)
        return self.o_proj(output)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class AfmoeTokenChoiceRouter(nn.Module):
    """Token-choice top-K router for MoE routing."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states, expert_bias: Optional[torch.Tensor]):
        _, _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        scores = self.gate(hidden_states)

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        else:
            scores = F.softmax(scores.to(torch.float32), dim=-1)

        if expert_bias is not None:
            _, selected_experts = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts)
        else:
            top_scores, selected_experts = torch.topk(scores, k=self.top_k, dim=1)

        if self.score_func == "sigmoid" and self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator

        top_scores = top_scores * self.route_scale
        return top_scores, selected_experts


# ---------------------------------------------------------------------------
# MoE with kt-kernel integration
# ---------------------------------------------------------------------------

class AfmoeMoE(nn.Module):
    """
    MoE layer adapted for kt-kernel CPU/GPU hybrid inference.

    - Routed experts are offloaded to CPU via KTMoEWrapper
    - Shared experts remain on GPU
    - During init, call `init_kt_moe()` after model loading to set up KTMoEWrapper
    """

    def __init__(self, config: AfmoeConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.router = AfmoeTokenChoiceRouter(config)

        # Shared experts (always on GPU)
        self.shared_experts = None
        if config.num_shared_experts > 0:
            self.shared_experts = AfmoeMLP(
                config, config.moe_intermediate_size * config.num_shared_experts
            )

        # Routed experts: keep nn.ModuleList for weight loading compatibility
        self.experts = nn.ModuleList(
            [
                AfmoeMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

        self.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32), requires_grad=False
        )

        # kt-kernel wrapper (initialized after weight loading)
        self.kt_moe_wrapper: Optional[KTMoEWrapper] = None
        self._use_kt_kernel = False

    def init_kt_moe(
        self,
        layer_idx: int,
        gpu_experts_mask: Optional[torch.Tensor] = None,
        cpuinfer_threads: int = 64,
        threadpool_count: int = 2,
        weight_path: str = "",
        chunked_prefill_size: int = 512,
        method: str = "BF16",
        max_deferred_experts_per_token: int = 0,
        numa_nodes=None,
    ):
        """
        Initialize KTMoEWrapper for this layer's routed experts.

        Call this AFTER model weights are loaded. For BF16/FP8 methods,
        weights are loaded from tensors directly; for AMX methods,
        weights are loaded from pre-quantized files on disk.

        Args:
            layer_idx: The MoE layer index (0-based among MoE layers)
            gpu_experts_mask: Boolean mask [num_experts], True = on GPU
            cpuinfer_threads: Number of CPU threads for inference
            threadpool_count: Number of NUMA subpools
            weight_path: Path to quantized weights (for AMX/LLAMAFILE methods)
            chunked_prefill_size: Prefill chunk size
            method: Backend method (BF16, FP8, AMXINT4, AMXINT8, LLAMAFILE, etc.)
            max_deferred_experts_per_token: Deferred experts for pipelining
            numa_nodes: NUMA node list
        """
        self.kt_moe_wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            method=method,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            numa_nodes=numa_nodes,
        )

        # For native methods (BF16/FP8), load weights from the expert tensors
        if method in ["BF16", "FP8", "FP8_PERCHANNEL"]:
            gate_proj = torch.stack(
                [self.experts[i].gate_proj.weight.data for i in range(self.num_experts)]
            )
            up_proj = torch.stack(
                [self.experts[i].up_proj.weight.data for i in range(self.num_experts)]
            )
            down_proj = torch.stack(
                [self.experts[i].down_proj.weight.data for i in range(self.num_experts)]
            )

            # physical_to_logical_map: identity mapping (expert i -> expert i)
            physical_to_logical_map = torch.arange(
                self.num_experts, dtype=torch.long, device="cpu"
            )

            self.kt_moe_wrapper.load_weights_from_tensors(
                gate_proj, up_proj, down_proj, physical_to_logical_map
            )
        else:
            # For AMX/LLAMAFILE: load from disk
            physical_to_logical_map = torch.arange(
                self.num_experts, dtype=torch.long, device="cpu"
            )
            self.kt_moe_wrapper.load_weights(physical_to_logical_map)

        self._use_kt_kernel = True

        # Free the nn.ModuleList expert weights to save GPU memory
        # (weights are now in kt-kernel's CPU buffers)
        self.experts = nn.ModuleList()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router: get topk expert ids and weights
        top_scores, selected_experts = self.router(hidden_states, self.expert_bias)
        # top_scores: [num_tokens, num_experts_per_tok]
        # selected_experts: [num_tokens, num_experts_per_tok]

        # Shared experts (always on GPU)
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat)
        else:
            shared_output = torch.zeros_like(hidden_states_flat)

        # Routed experts via kt-kernel
        if self._use_kt_kernel:
            cuda_stream = torch.cuda.current_stream().cuda_stream
            routed_output = self.kt_moe_wrapper.forward(
                hidden_states_flat,
                selected_experts,
                top_scores,
                cuda_stream,
            )
        else:
            # Fallback: original PyTorch implementation (for debugging/no kt-kernel)
            routed_output = self._pytorch_moe_forward(
                hidden_states_flat, top_scores, selected_experts
            )

        # Combine shared + routed
        output = shared_output + routed_output
        return output.view(batch_size, seq_len, hidden_dim)

    def _pytorch_moe_forward(
        self,
        hidden_states_flat: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback PyTorch MoE forward (no kt-kernel)."""
        hidden_dim = hidden_states_flat.shape[-1]

        # Sort tokens by expert
        token_indices_sorted = torch.argsort(selected_experts.view(-1), stable=True)
        top_scores_sorted = top_scores.view(-1)[token_indices_sorted]
        token_to_expert = selected_experts.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.num_experts_per_tok

        token_indices_expanded = token_indices_sorted.unsqueeze(-1).expand(-1, hidden_dim)
        routed_input = torch.gather(
            hidden_states_flat, dim=0, index=token_indices_expanded
        )

        routed_output = torch.zeros_like(routed_input)
        for expert_id in range(self.num_experts):
            mask = token_to_expert == expert_id
            if mask.any():
                expert_input = routed_input[mask]
                expert_out = self.experts[expert_id](expert_input)
                routed_output[mask] = expert_out

        routed_output = (
            routed_output.to(torch.float32) * top_scores_sorted.unsqueeze(-1)
        ).to(hidden_states_flat.dtype)

        # Scatter back
        output = torch.zeros_like(hidden_states_flat)
        output.scatter_add_(dim=0, index=token_indices_expanded, src=routed_output)
        return output


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class AfmoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = AfmoeAttention(config=config, layer_idx=layer_idx)
        self.attention_type = config.layer_types[layer_idx]

        self.input_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AfmoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_mlp_layernorm = AfmoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_mlp_layernorm = AfmoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.moe_enabled = layer_idx >= config.num_dense_layers
        if self.moe_enabled:
            self.mlp = AfmoeMoE(config)
        else:
            self.mlp = AfmoeMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        # Self Attention with dual normalization
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # FFN with dual normalization
        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AfmoePreTrainedModel(PreTrainedModel):
    config_class = AfmoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["AfmoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _keep_in_fp32_modules = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_mlp_layernorm",
        "post_mlp_layernorm",
        "q_norm",
        "k_norm",
        "norm",
    ]
    _supports_sdpa = True
    _supports_attention_backend = True
    supports_gradient_checkpointing = True


class AfmoeModel(AfmoePreTrainedModel):
    _no_split_modules = ["AfmoeDecoderLayer"]

    def __init__(self, config: AfmoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                AfmoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = AfmoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def init_kt_kernel(
        self,
        gpu_experts_mask_per_layer: Optional[dict] = None,
        cpuinfer_threads: int = 64,
        threadpool_count: int = 2,
        weight_path: str = "",
        chunked_prefill_size: int = 512,
        method: str = "BF16",
        max_deferred_experts_per_token: int = 0,
        num_gpu_experts: int = 0,
        numa_nodes=None,
    ):
        """
        Initialize kt-kernel for all MoE layers in this model.

        Args:
            gpu_experts_mask_per_layer: Optional dict {moe_layer_idx: mask_tensor}.
                If None, uses uniform distribution based on num_gpu_experts.
            cpuinfer_threads: CPU inference threads
            threadpool_count: NUMA subpool count
            weight_path: Path to quantized weights
            chunked_prefill_size: Prefill chunk size
            method: Backend method
            max_deferred_experts_per_token: Deferred experts count
            num_gpu_experts: Number of GPU experts per layer (uniform)
            numa_nodes: NUMA nodes list
        """
        moe_layer_idx = 0
        for layer in self.layers:
            if hasattr(layer, "mlp") and isinstance(layer.mlp, AfmoeMoE):
                if gpu_experts_mask_per_layer and moe_layer_idx in gpu_experts_mask_per_layer:
                    mask = gpu_experts_mask_per_layer[moe_layer_idx]
                elif num_gpu_experts > 0:
                    mask = torch.zeros(
                        self.config.num_experts, dtype=torch.bool
                    )
                    # Uniform: place first N experts on GPU
                    mask[:num_gpu_experts] = True
                else:
                    mask = None

                layer.mlp.init_kt_moe(
                    layer_idx=moe_layer_idx,
                    gpu_experts_mask=mask,
                    cpuinfer_threads=cpuinfer_threads,
                    threadpool_count=threadpool_count,
                    weight_path=weight_path,
                    chunked_prefill_size=chunked_prefill_size,
                    method=method,
                    max_deferred_experts_per_token=max_deferred_experts_per_token,
                    numa_nodes=numa_nodes,
                )
                moe_layer_idx += 1

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        if self.config.mup_enabled:
            hidden_states = hidden_states * (self.config.hidden_size**0.5)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class AfmoeForCausalLM(AfmoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = AfmoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def init_kt_kernel(self, **kwargs):
        """Convenience: delegates to self.model.init_kt_kernel()."""
        self.model.init_kt_kernel(**kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "AfmoeForCausalLM",
    "AfmoeModel",
    "AfmoePreTrainedModel",
]
