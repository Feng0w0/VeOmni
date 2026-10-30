"""Tensor contract shared by QSA compressed-block selectors."""

import torch


def validate(q, k, *, top_k, compress_ratio, query_offset):
    """Validate one causal segment per batch row, with preprocessed compressed keys."""
    if q.ndim != 4 or k.ndim != 3 or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA indexer expects q=[B,Q,H,D] and k=[B,K,D].")
    if q.device != k.device or q.dtype != k.dtype or not q.is_floating_point():
        raise ValueError("QSA q/k must have the same floating dtype and device.")
    if min(q.shape[0], q.shape[2], q.shape[3], top_k, compress_ratio) <= 0 or query_offset < 0:
        raise ValueError("QSA requires positive batch/heads/dim/top_k/compress_ratio and query_offset >= 0.")
    if k.shape[1] < (query_offset + q.shape[1]) // compress_ratio:
        raise ValueError("QSA compressed keys do not cover all complete blocks through the last query.")


def dense_prefix(q, *, top_k, compress_ratio, query_offset):
    """Return deterministic block IDs for rows that can see at most top_k blocks."""
    rows = min(q.shape[1], max(0, (top_k + 1) * compress_ratio - 1 - query_offset))
    ids = torch.arange(top_k, dtype=torch.int32, device=q.device)
    counts = (query_offset + torch.arange(rows, device=q.device) + 1) // compress_ratio
    result = ids.view(1, 1, -1).expand(q.shape[0], rows, -1)
    return result.masked_fill(ids.view(1, 1, -1) >= counts.view(1, -1, 1), -1)
