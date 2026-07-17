#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v24 safe baseline: 0.8304 base + 0.8311 time-bias trick, no item FAFE ----
# Sequences are stored newest-first; LongerEncoder._gather_top_k has been
# rewritten to take the first top_k valid tokens (model.py).
# Per EDA (eda/.../sequence_summary.csv):
#   seq_d  mean_len=2457.6  cnt_1d/row=74.26  cnt_7d/row=400.24
#   seq_a/b/c have mean_len < 800 and last_p50 >= 9h — no need to compress.
# top_k=128 covers ~1.5-2 days for the median seq_d user; cross-attn cost
# drops to ~30% of the dense-Transformer-512 baseline for that domain.
# Training forward uses torch.compile by default for speed.
# Validation predict stays in eager mode unless --compile_eval is explicitly set.
# Inference/test also stays eager for reproducibility and lower graph-break risk.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --seq_encoder_type transformer \
    --longer_domains seq_d \
    --seq_top_k 128 \
    --compile \
    --compile_eval \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
