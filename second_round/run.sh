#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ── GPU detection ──
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES// /}" ]]; then
    NGPUS=0
  else
    NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
  fi
elif command -v nvidia-smi &> /dev/null; then
  NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  NGPUS=0
fi
echo "Using ${NGPUS} GPU(s) (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}')"

# ── Training args ──
TRAIN_ARGS=(
    --ns_tokenizer_type rankmixer
    --user_ns_tokens 4
    --item_ns_tokens 2
    --num_queries 2
    --ns_groups_json ""

    # v29 final: 100% data (no validation holdout), train exactly 8 epochs,
    # save a self-contained checkpoint after EVERY epoch (epochN_step*/model.pt).
    # No val => no early stopping; pick which epoch to submit yourself.
    --valid_ratio 0
    --num_epochs 8
    --save_every_epoch
    --emb_skip_threshold 1000000
    --seq_hash_bucket_size 500000
    --ns_hash_bucket_size 500000
    --num_workers 8
    --seq_encoder_type swiglu

    # v43: item_dense now emits 2 tokens. num_ns=10, T=2*4+10=18 in
    # rank_mixer full mode, so d_model must be divisible by both 18 and heads=5.
    --d_model 108
    --emb_dim 64
    --num_hyformer_blocks 2
    --num_heads 6
    --hidden_mult 6
    --batch_size 1024
    --seq_top_k 50

    # EMA over dense params only (sparse embeddings excluded). Eval & saved
    # checkpoint use the averaged weights. 0 disables.
    --ema_decay 0.9995

    # v23: supervised contrastive (SupCon) aux loss on the final representation.
    # v23 change vs v22: only the positive class (converted samples) act as
    # anchors (--supcon_pos_anchor_only), so SupCon pulls the rare conversions
    # together instead of the ill-posed "pull all non-converted together".
    # Ablations: drop --use_supcon (=v20); add --no_supcon_pos_anchor_only (=v22).
    --use_supcon
    --supcon_weight 0.1
    --supcon_temp 0.1
    --supcon_pos_anchor_only

    # v25: DIN-MLP target-aware weighted pooling in the query generator
    # (ported from v17). Replaces the mean pooling that summarizes each
    # sequence into the Q-generator input. Ablation back to v24: drop this line.
    --query_pool_mode target_attn

    # v28/v29: word2vec warm-start (gensim, in-job) + FROZEN (--seq_w2v_freeze)
    # for the selected high-card seq ids. The embeddings live in the single
    # model.pt; no external w2v artifact at inference.
    # v46: expand the word2vec set to the v43/final 7 seq ids plus the
    # remaining seq hash field seq_c/29. The non-seq hash field user_int/116
    # stays on NS hash embedding because seq_w2v only consumes seq side-info.
    # Selected ids (the rest keep v27 handling = hash/direct embedding):
    #   seq_a/38, seq_b/69, seq_c/29, seq_c/34, seq_c/36, seq_c/47,
    #   seq_d/22, seq_d/23
    # Ablation back to v27 (hash/direct for all): drop --use_seq_w2v.
    --use_seq_w2v
    --seq_w2v_ids seq_a:38,seq_b:69,seq_c:29,seq_c:34,seq_c:36,seq_c:47,seq_d:22,seq_d:23
    --seq_w2v_top_k 2000000
    --seq_w2v_min_count 5
    --seq_w2v_epochs 3
    --seq_w2v_max_sentences 2000000
    --seq_w2v_freeze

    # v46: NS-side word2vec for the remaining high-card NS hash field.
    # This replaces user_int/116's NS hash embedding; item_int has no
    # high-card field above emb_skip_threshold.
    --use_ns_w2v
    --ns_w2v_ids user:116
    --ns_w2v_freeze
)

# ── Launch ──
if [[ "${NGPUS}" -gt 1 ]]; then
  echo "Detected ${NGPUS} GPUs, launching DDP with torchrun"
  torchrun --standalone --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train.py" "${TRAIN_ARGS[@]}" "$@"
else
  echo "Single GPU / CPU mode"
  python3 -u "${SCRIPT_DIR}/train.py" "${TRAIN_ARGS[@]}" "$@"
fi

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, replace TRAIN_ARGS above with:
#
# TRAIN_ARGS=(
#     --ns_tokenizer_type group
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json"
#     --num_queries 1
#     --emb_skip_threshold 1000000
#     --num_workers 8
# )
