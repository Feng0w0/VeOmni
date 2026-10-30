# Qwen4-Exp QSA AscendC indexer

`Qwen4ExpTextQSAIndexer.forward` calls
`veomni.ops.kernels.qsa_indexer.ascendc.qsa_indexer.qwen4_exp_qsa_indexer_forward_lightning` at the start of forward when
`hidden_states.device.type == "npu"`. CPU/CUDA retain the original eager scoring
and selection code in the model. This package registers no VeomniOp and adds no
`qsa_indexer_implementation` configuration field.

The AscendC forward owns projection, normalization, partial RoPE, packed sample boundaries,
compressed-key gathering for Ulysses/CP, global token translation and causal tail
tokens. The kernel receives one segment per row: Q `[B,Q,H,D]`, compressed K
`[B,K,D]`, and a segment-relative `query_offset`; it returns block IDs
`[B,Q,top_k]`, int32 with -1 padding. The model returns compact global token IDs,
not a dense token mask. KV/indexer cache remains unsupported.

Requires CANN `cann_ops_transformer`, NPU FP16/BF16, D=128, 1<=H<=64,
compression ratio 1/4/128 and top_k<=8192. The kernel zero-pads heads to 32/64,
uses right-down causal metadata and excludes future blocks on non-final SP ranks.
Unsupported inputs raise; there is no silent eager fallback. CANN is imported
only when actual kernel work is required. A dense-prefix fast path selects all
visible blocks when their count does not exceed the budget.

Only the discrete selection kernel uses `torch.no_grad`; the model forward is
not globally decorated. Selection remains nondifferentiable. Top-k ties and
floating-point accumulation may select different blocks than eager at a cutoff.

Source: MindSpeed-MM_3047/qwen38_1, mindspeed_mm/fsdp/ops/qsa/ascendc/qsa_indexer.py.
Implementation is adapted to VeOmni's packed/SP compact-index contract.
