"""QSA model direct AscendC dispatch and original eager packed selection."""

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils.device import IS_NPU_AVAILABLE


def make_indexer(device):
    """Use exactly separated block scores to check index layout without cutoff ties."""
    projection = torch.nn.Linear(2, 5 * 128, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        projection.weight.zero_()
        for head in range(4):
            projection.weight[head * 128, 0] = 1
        projection.weight[4 * 128, 1] = 1
    return SimpleNamespace(
        index_kv_heads=1,
        index_n_heads=4,
        index_head_dim=128,
        compress_ratio=4,
        block_topk=4,
        token_budget=16,
        index_qk_proj=projection,
        q_layernorm=torch.nn.Identity(),
        k_layernorm=torch.nn.Identity(),
        veomni_rope=lambda q, k, cos, sin, **kwargs: (q, k),
    )


def packed_inputs(device, lengths):
    """Return two 64-token rows, including a compression block across the SP cut."""
    hidden = torch.ones(2, 64, 2, dtype=torch.bfloat16, device=device)
    hidden[:, :, 1] = torch.arange(64, device=device) / 4 + 1
    positions = (torch.ones(2, 64, 128, device=device), torch.zeros(2, 64, 128, device=device))
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    return hidden, positions, boundaries


def expected_tokens(lengths):
    """Independently enumerate the newest complete blocks plus each causal tail."""
    output = torch.full((2, 64, 19), -1, dtype=torch.int32)
    flat_start = 0
    for length in lengths:
        batch, start = divmod(flat_start, 64)
        for offset in range(length):
            complete = (offset + 1) // 4
            ids = list(range(start + max(0, complete - 4) * 4, start + complete * 4))
            ids += list(range(start + complete * 4, start + offset + 1))
            output[batch, start + offset, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        flat_start += length
    return output


@pytest.mark.parametrize(
    "implementation",
    [
        pytest.param("eager", marks=pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))),
        pytest.param(
            "ascendc",
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE or find_spec("cann_ops_transformer") is None, reason="requires Ascend and CANN"
            ),
        ),
    ],
)
@pytest.mark.parametrize("lengths", [(64, 64), (9, 31, 24, 35, 29), (1, 2, 61, 3, 61)])
def test_packed_model_indexer(implementation, lengths):
    from veomni.models.transformers.qwen4_exp.generated import patched_modeling_qwen4_exp_gpu as modeling

    device = "npu" if implementation == "ascendc" else "cpu"
    hidden, positions, boundaries = packed_inputs(device, lengths)
    actual = modeling.Qwen4ExpTextQSAIndexer.forward(
        make_indexer(device),
        hidden,
        positions,
        None,
        None,
        cu_seq_lens_q=boundaries,
        packed_seq_lens=lengths,
    )
    torch.testing.assert_close(actual.cpu().sort(-1).values, expected_tokens(lengths).sort(-1).values)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_ascendc_dispatch_precedes_preprocessing(monkeypatch):
    from importlib import import_module

    from veomni.models.transformers.qwen4_exp.generated import patched_modeling_qwen4_exp_gpu as modeling

    backend = import_module("veomni.ops.kernels.qsa_indexer.ascendc.qsa_indexer")
    hidden = SimpleNamespace(device=SimpleNamespace(type="npu"))
    indexer, positions, mask, cache, boundaries = (object() for _ in range(5))
    lengths = (9, 31)
    result = object()
    calls = []

    def forward(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(backend, "qwen4_exp_qsa_indexer_forward_lightning", forward)
    actual = modeling.Qwen4ExpTextQSAIndexer.forward(
        indexer,
        hidden,
        positions,
        mask,
        cache,
        cu_seq_lens_q=boundaries,
        packed_seq_lens=lengths,
    )
    assert actual is result
    assert calls == [
        ((indexer, hidden, positions, mask, cache), {"cu_seq_lens_q": boundaries, "packed_seq_lens": lengths})
    ]
