"""Synthetic, exactly balanced expert routing for performance experiments."""

import math

import torch
from torch import Tensor


def force_balanced_routing(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    num_experts: int,
    ep_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return padded flat inputs, slot weights and deterministic expert IDs.

    Every source contributes equally to every expert, so destination EP ranks
    receive equal totals even with different source token counts. Real slot
    weights retain their gradients, but their original expert choices are
    deliberately replaced. Dummy tokens have zero inputs and zero weights.
    """
    if ep_size < 1 or num_experts < 1 or num_experts % ep_size:
        raise ValueError("num_experts must be positive and divisible by ep_size")
    if routing_weights.ndim != 2 or selected_experts.shape != routing_weights.shape:
        raise ValueError("routing weights and expert IDs must have matching [tokens, topk] shapes")
    tokens, topk = routing_weights.shape
    if not 1 <= topk <= num_experts:
        raise ValueError("topk must be between 1 and num_experts")
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if flat.shape[0] != tokens:
        raise ValueError("hidden_states and routing_weights must have the same token count")
    if not (flat.device == routing_weights.device == selected_experts.device):
        raise ValueError("inputs must be on the same device")
    # Make padded_tokens * topk divisible by E without duplicate experts in
    # any token's top-k. Keep empty sources valid for kernels requiring work.
    quantum = num_experts // math.gcd(num_experts, topk)
    padded_tokens = max(quantum, ((tokens + quantum - 1) // quantum) * quantum)
    padding = padded_tokens - tokens
    if padding:
        flat = torch.cat((flat, flat.new_zeros(padding, flat.shape[-1])))
        routing_weights = torch.cat((routing_weights, routing_weights.new_zeros(padding, topk)))
    slots = torch.arange(padded_tokens * topk, device=flat.device, dtype=torch.int64) % num_experts
    # Interleave destination ranks while retaining contiguous expert ownership.
    experts = (slots % ep_size) * (num_experts // ep_size) + slots // ep_size
    return flat, routing_weights, experts.reshape(padded_tokens, topk).to(selected_experts.dtype)
