"""CANN LightningIndexerV2 selection adapted from MindSpeed-MM's QSA indexer.

Source: songke029/MindSpeed-MM_3047, qwen38_1,
mindspeed_mm/fsdp/ops/qsa/ascendc/qsa_indexer.py.
The complete forward below adapts projection, compression, RoPE, packed layout
and sequence-parallel communication to VeOmni. CANN imports are lazy.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

from veomni.distributed.context_parallel.dsa_cp import all_gather_compressed_rows
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs

from .common import dense_prefix, validate


def _ops():
    from cann_ops_transformer import ops

    return ops


@lru_cache(maxsize=8)
def _metadata(batch, rows, blocks, heads, dim, top_k, ratio, remainder, device):
    residual = torch.full((batch,), remainder, dtype=torch.int32, device=device)
    metadata = _ops().lightning_indexer_metadata(
        heads,
        1,
        dim,
        top_k,
        cmp_residual_k=residual,
        batch_size=batch,
        max_seqlen_q=rows,
        max_seqlen_k=blocks,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    return metadata, residual


@torch.no_grad()
def qsa_indexer(q, k, *, top_k, compress_ratio, query_offset=0):
    """Return int32 [B,Q,top_k] causal block IDs using LightningIndexerV2.

    Heads below 32 (including Qwen4-Exp's four heads) are zero-padded to 32;
    heads between 33 and 64 are padded to 64. Zero heads do not change the
    sum of ReLU scores. Unsupported inputs raise without an eager fallback.
    """
    validate(q, k, top_k=top_k, compress_ratio=compress_ratio, query_offset=query_offset)
    batch, rows, heads, dim = q.shape
    if q.device.type != "npu" or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("CANN QSA indexer requires NPU float16/bfloat16 q/k.")
    if dim != 128 or heads > 64 or compress_ratio not in (1, 4, 128) or top_k > 8192:
        raise ValueError("CANN QSA requires D=128, 1<=H<=64, ratio in {1,4,128}, and top_k<=8192.")
    prefix = dense_prefix(q, top_k=top_k, compress_ratio=compress_ratio, query_offset=query_offset)
    dense_rows = prefix.shape[1]
    if dense_rows == rows:
        return prefix.contiguous()

    # Right-down causality aligns Q to the *end* of the pre-compression keys.
    # Crop future blocks for non-final SP ranks before building the metadata.
    key_end = query_offset + rows
    blocks = key_end // compress_ratio
    kernel_heads = 32 if heads <= 32 else 64
    query = F.pad(q[:, dense_rows:], (0, 0, 0, kernel_heads - heads)).contiguous()
    # CANN checks strides even on singleton axes; contiguous() alone can
    # retain a larger batch stride on a B=1 sliced view.
    keys = k[:, :blocks].clone(memory_format=torch.contiguous_format).unsqueeze(2)
    weights = torch.full((batch, rows - dense_rows, kernel_heads), dim**-0.5, dtype=torch.float32, device=q.device)
    metadata, residual = _metadata(
        batch, rows - dense_rows, blocks, kernel_heads, dim, top_k, compress_ratio, key_end % compress_ratio, q.device
    )
    ids, _ = _ops().lightning_indexer(
        query,
        keys,
        weights,
        top_k,
        cmp_residual_k=residual,
        metadata=metadata,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=compress_ratio,
        return_value=0,
    )
    return torch.cat((prefix, ids.squeeze(2).to(torch.int32)), dim=1).contiguous()


def _qwen4_exp_sequence_parallel_state(parallel_state):
    """Resolve the active sequence-parallel group/rank/size across Ulysses and CP."""
    ulysses_enabled = bool(getattr(parallel_state, "ulysses_enabled", False))
    cp_enabled = bool(getattr(parallel_state, "cp_enabled", False))
    if ulysses_enabled and cp_enabled:
        raise NotImplementedError(
            "Qwen4-Exp does not support simultaneous Ulysses and Context Parallelism (ulysses_size > 1 and cp_size > 1)."
        )
    if cp_enabled:
        return (True, parallel_state.cp_group, parallel_state.cp_rank, parallel_state.cp_size)
    if ulysses_enabled:
        return (True, parallel_state.ulysses_group, parallel_state.ulysses_rank, parallel_state.ulysses_size)
    return (False, None, 0, 1)


def qwen4_exp_qsa_indexer_forward_lightning(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    cu_seq_lens_q: torch.Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Complete VeOmni AscendC forward, including packed/SP layout and compact token output."""
    del attention_mask
    parallel_state = get_parallel_state()
    if past_key_values is not None:
        raise NotImplementedError("Qwen4-Exp compact QSA does not support a KV/indexer cache.")
    _, group, rank, world_size = _qwen4_exp_sequence_parallel_state(parallel_state)
    batch_size, local_seq_len, _ = hidden_states.shape
    global_seq_len = local_seq_len * world_size
    if self.index_kv_heads != 1:
        raise ValueError(f"Qwen4-Exp compact QSA currently requires one indexer KV head; got {self.index_kv_heads}.")
    if self.compress_ratio > local_seq_len:
        raise ValueError(
            f"Qwen4-Exp compact QSA needs each sequence-parallel shard to be at least one compression block wide; got local_seq_len={local_seq_len}, compress_ratio={self.compress_ratio}."
        )
    qk = self.index_qk_proj(hidden_states)
    q_width = self.index_n_heads * self.index_head_dim
    q, raw_keys = torch.split(qk, [q_width, self.index_head_dim], dim=-1)
    q = self.q_layernorm(q.reshape(batch_size, local_seq_len, self.index_n_heads, self.index_head_dim))
    local_cos, local_sin = position_embeddings
    if local_cos.shape[1] != local_seq_len or local_sin.shape[1] != local_seq_len:
        raise ValueError(
            f"Qwen4-Exp compact QSA position embeddings must match the local sequence length; got cos={local_cos.shape[1]}, sin={local_sin.shape[1]}, local_seq_len={local_seq_len}."
        )
    local_start = rank * local_seq_len
    q, _ = self.veomni_rope(q, q, local_cos, local_sin, unsqueeze_dim=2)
    if cu_seq_lens_q is None:
        segments = [(batch_idx, 0, global_seq_len) for batch_idx in range(batch_size)]
    else:
        if packed_seq_lens is None:
            boundaries = [int(value) for value in cu_seq_lens_q.tolist()]
        else:
            boundaries = [0]
            for length in packed_seq_lens:
                boundaries.append(boundaries[-1] + length)
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != batch_size * global_seq_len:
            raise ValueError(
                f"Qwen4-Exp compact QSA requires cu_seq_lens_q to cover the complete padded batch; got endpoints {boundaries[:1]}..{boundaries[-1:]}, expected 0..{batch_size * global_seq_len}."
            )
        segments = []
        for flat_start, flat_end in zip(boundaries, boundaries[1:]):
            if flat_end <= flat_start:
                raise ValueError("Qwen4-Exp compact QSA requires strictly increasing cu_seq_lens_q.")
            batch_idx = flat_start // global_seq_len
            if (flat_end - 1) // global_seq_len != batch_idx:
                raise ValueError("Qwen4-Exp compact QSA packed samples may not cross the batch dimension.")
            segments.append(
                (batch_idx, flat_start - batch_idx * global_seq_len, flat_end - batch_idx * global_seq_len)
            )
    segment_ids = torch.empty((batch_size, global_seq_len), dtype=torch.long, device=hidden_states.device)
    segment_starts = torch.empty_like(segment_ids)
    for segment_id, (batch_idx, start, end) in enumerate(segments):
        segment_ids[batch_idx, start:end] = segment_id
        segment_starts[batch_idx, start:end] = start
    blocks = [
        (segment_id, batch_idx, start)
        for segment_id, (batch_idx, segment_start, segment_end) in enumerate(segments)
        for start in range(segment_start, segment_end - self.compress_ratio + 1, self.compress_ratio)
    ]
    owned_blocks = [block for block in blocks if block[2] // local_seq_len == rank]
    gathered_blocks = [block for owner in range(world_size) for block in blocks if block[2] // local_seq_len == owner]
    count_list = [
        sum((block_start // local_seq_len == owner for _, _, block_start in blocks)) for owner in range(world_size)
    ]
    local_batch_id_list = [batch_idx for _, batch_idx, _ in owned_blocks]
    local_block_start_list = [start for _, _, start in owned_blocks]
    gathered_segment_id_list = [segment_id for segment_id, _, _ in gathered_blocks]
    gathered_start_list = [start for _, _, start in gathered_blocks]
    meta_sections = (
        count_list,
        local_batch_id_list,
        local_block_start_list,
        gathered_segment_id_list,
        gathered_start_list,
    )
    flat_meta = torch.tensor([value for section in meta_sections for value in section], dtype=torch.long).to(
        hidden_states.device
    )
    counts_end = len(count_list)
    local_ids_end = counts_end + len(local_batch_id_list)
    local_starts_end = local_ids_end + len(local_block_start_list)
    gathered_seg_ids_end = local_starts_end + len(gathered_segment_id_list)
    counts = flat_meta[:counts_end]
    local_batch_ids = flat_meta[counts_end:local_ids_end]
    local_block_starts = flat_meta[local_ids_end:local_starts_end]
    block_starts = flat_meta[gathered_seg_ids_end:]
    halo = self.compress_ratio - 1
    if halo == 0 or world_size == 1:
        right_halo = raw_keys[:, :0]
    else:
        prefixes = gather_outputs(raw_keys[:, :halo].contiguous(), gather_dim=1, group=group)
        if rank + 1 == world_size:
            right_halo = prefixes[:, :halo] * 0
        else:
            halo_start = (rank + 1) * halo
            right_halo = prefixes[:, halo_start : halo_start + halo]
    extended_keys = torch.cat((raw_keys, right_halo), dim=1)
    if owned_blocks:
        local_offsets = local_block_starts - local_start
        window_indices = local_offsets[:, None] + torch.arange(self.compress_ratio, device=hidden_states.device)[None]
        pooled_keys = extended_keys[local_batch_ids[:, None], window_indices].float().mean(dim=1).to(raw_keys.dtype)
        pooled_keys = self.k_layernorm(pooled_keys)
        pooled_keys = pooled_keys[None, None]
        pooled_keys, _ = self.veomni_rope(
            pooled_keys,
            pooled_keys,
            local_cos[local_batch_ids, local_offsets][None],
            local_sin[local_batch_ids, local_offsets][None],
        )
        pooled_keys = pooled_keys[0]
    else:
        pooled_keys = self.k_layernorm(extended_keys[:, :0]).reshape(1, 0, self.index_head_dim)
    if world_size > 1:
        pooled_keys = all_gather_compressed_rows(pooled_keys, counts, group)
    output_width = self.token_budget + self.compress_ratio - 1
    query_positions = local_start + torch.arange(local_seq_len, device=hidden_states.device)
    local_segment_starts = segment_starts[:, local_start : local_start + local_seq_len]
    selected_local = torch.full(
        (batch_size, local_seq_len, self.block_topk * self.compress_ratio),
        -1,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    block_offsets = torch.arange(self.compress_ratio, dtype=torch.int32, device=hidden_states.device)
    for segment_id, (batch_idx, segment_start, segment_end) in enumerate(segments):
        query_start = max(segment_start, local_start)
        query_end = min(segment_end, local_start + local_seq_len)
        if query_start >= query_end:
            continue
        block_indices = [i for i, (sid, _, _) in enumerate(gathered_blocks) if sid == segment_id]
        if not block_indices:
            continue
        if block_indices[-1] - block_indices[0] + 1 == len(block_indices):
            segment_keys = pooled_keys[:, block_indices[0] : block_indices[-1] + 1]
            segment_block_starts = block_starts[block_indices[0] : block_indices[-1] + 1]
        else:
            block_indices_tensor = torch.tensor(block_indices, device=hidden_states.device)
            segment_keys = pooled_keys.index_select(1, block_indices_tensor)
            segment_block_starts = block_starts.index_select(0, block_indices_tensor)
        local_begin, local_end = (query_start - local_start, query_end - local_start)
        ids = qsa_indexer(
            q[batch_idx : batch_idx + 1, local_begin:local_end],
            segment_keys,
            top_k=self.block_topk,
            compress_ratio=self.compress_ratio,
            query_offset=query_start - segment_start,
        )[0]
        starts = segment_block_starts[ids.clamp_min(0).long()].to(torch.int32)
        tokens = starts[..., None] + block_offsets
        tokens = tokens.masked_fill(ids[..., None] < 0, -1).flatten(-2)
        selected_local[batch_idx, local_begin:local_end] = tokens
    visible_count = query_positions[None] - local_segment_starts + 1
    tail_count = torch.remainder(visible_count, self.compress_ratio)
    tail_start = (
        local_segment_starts
        + torch.div(visible_count, self.compress_ratio, rounding_mode="floor") * self.compress_ratio
    )
    tail_offsets = torch.arange(self.compress_ratio - 1, device=hidden_states.device)
    tails = tail_start[..., None] + tail_offsets
    tails = tails.masked_fill(tail_offsets[None, None] >= tail_count[..., None], -1)
    selected_local = torch.cat((selected_local, tails), dim=-1)
    if selected_local.shape[-1] < output_width:
        selected_local = torch.nn.functional.pad(
            selected_local, (0, output_width - selected_local.shape[-1]), value=-1
        )
    selected_local = selected_local[..., :output_width].to(torch.int32).contiguous()
    return selected_local
