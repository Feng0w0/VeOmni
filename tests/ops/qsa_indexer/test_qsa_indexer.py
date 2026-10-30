"""Causal block selection, backend requirements and real CANN numerical parity."""

from importlib.util import find_spec

import pytest
import torch

from veomni.ops.kernels.qsa_indexer.ascendc.qsa_indexer import qsa_indexer
from veomni.utils.device import IS_NPU_AVAILABLE


def _reference(q, k, top_k, ratio, offset):
    result = torch.full((*q.shape[:2], top_k), -1, dtype=torch.int32)
    for batch in range(q.shape[0]):
        for row in range(q.shape[1]):
            count = (offset + row + 1) // ratio
            scores = (q[batch, row].double() @ k[batch, :count].double().T).relu().sum(0)
            selected = scores.argsort(descending=True)[:top_k]
            result[batch, row, : selected.numel()] = selected.int()
    return result


@pytest.mark.skipif(
    not IS_NPU_AVAILABLE or find_spec("cann_ops_transformer") is None, reason="requires Ascend and CANN"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "ratio,offset,rows,top_k",
    [(1, 0, 37, 4), (4, 0, 79, 4), (4, 29, 37, 4), (4, 41, 23, 4), (128, 121, 512, 2), (4, 0, 3, 4)],
)
def test_cann_exact_separated_scores(dtype, ratio, offset, rows, top_k):
    # Exactly representable, separated scores establish exact selection parity
    # without assuming identical tie-breaking or BF16 matmul accumulation.
    q = torch.zeros(2, rows, 4, 128, dtype=dtype)
    q[..., 0] = 1
    k = torch.zeros(2, (offset + rows) // ratio + 9, 128, dtype=dtype)
    k[..., 0] = torch.arange(k.shape[1]).to(dtype) / 4 + 1
    expected = _reference(q, k, top_k, ratio, offset)
    actual = qsa_indexer(q.npu(), k.npu(), top_k=top_k, compress_ratio=ratio, query_offset=offset).cpu()
    torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values)


@pytest.mark.skipif(
    not IS_NPU_AVAILABLE or find_spec("cann_ops_transformer") is None, reason="requires Ascend and CANN"
)
def test_cann_random_scores():
    torch.manual_seed(338)
    q = torch.randn(1, 257, 4, 128, dtype=torch.bfloat16)
    k = torch.randn(1, 100, 128, dtype=torch.bfloat16)
    actual = qsa_indexer(q.npu(), k.npu(), top_k=8, compress_ratio=4, query_offset=37).cpu().long()
    scores = torch.einsum("bqhd,bkd->bqkh", q.float(), k.float()).relu().sum(-1)
    count = (torch.arange(q.shape[1]) + 38) // 4
    visible = torch.arange(k.shape[1])[None, None] < count[None, :, None]
    assert ((actual < count[None, :, None]) | (actual == -1)).all()
    assert (actual >= 0).sum(-1).equal(count.clamp(max=8)[None])
    chosen = scores.gather(-1, actual.clamp_min(0)).masked_fill(actual < 0, -1)
    expected = scores.masked_fill(~visible, -1).topk(8, dim=-1).values
    torch.testing.assert_close(chosen.sort(-1).values, expected.sort(-1).values, rtol=0.02, atol=0.03)


@pytest.mark.skipif(
    not IS_NPU_AVAILABLE or find_spec("cann_ops_transformer") is None, reason="requires Ascend and CANN"
)
@pytest.mark.parametrize("heads,rows,top_k", [(4, 4099, 512), (32, 79, 4), (33, 79, 4), (64, 79, 4)])
def test_cann_model_budget_and_head_layouts(heads, rows, top_k):
    torch.manual_seed(448)
    q = torch.randn(1, rows, heads, 128, dtype=torch.bfloat16)
    k = torch.randn(1, rows // 4, 128, dtype=torch.bfloat16)
    actual = qsa_indexer(q.npu(), k.npu(), top_k=top_k, compress_ratio=4).cpu().long()
    counts = (torch.arange(rows) + 1) // 4
    assert ((actual < counts[None, :, None]) | (actual == -1)).all()
    assert (actual >= 0).sum(-1).equal(counts.clamp(max=top_k)[None])
    for row in [min(rows - 1, (top_k + 1) * 4), rows - 1]:
        selected = actual[0, row]
        selected = selected[selected >= 0]
        assert selected.unique().numel() == selected.numel()
        scores = (q[0, row].float() @ k[0, : counts[row]].float().T).relu().sum(0)
        expected = scores.topk(min(top_k, scores.numel())).values
        torch.testing.assert_close(scores[selected].sort().values, expected.sort().values, rtol=0.02, atol=0.03)
