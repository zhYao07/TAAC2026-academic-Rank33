"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    time_feats: torch.Tensor  # tensor [B, 3] = hour_id, weekday_id, period_id
    seq_time_feats: dict      # {domain: tensor [B, L, 3]}
    seq_day_type_feats: dict  # {domain: tensor [B, L, 2] = is_weekend, is_holiday}


NUM_CALENDAR_HOUR_IDS = 25
NUM_CALENDAR_WEEKDAY_IDS = 8
NUM_CALENDAR_PERIOD_IDS = 7
USER_CALENDAR_INT_VOCABS = (
    NUM_CALENDAR_HOUR_IDS - 1,
    NUM_CALENDAR_WEEKDAY_IDS - 1,
    NUM_CALENDAR_PERIOD_IDS - 1,
)
USER_CALENDAR_DENSE_DIM = 4
USER_CALENDAR_DENSE_FID_BASE = 990000


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class W2VSeqEmbedding(nn.Module):
    """Compressed-vocab embedding for a high-cardinality sequence id (v28).

    Replaces the hash / huge direct embedding for the main id columns. Holds:
      - ``kept_ids``: a sorted int64 buffer of the K kept raw ids (the most
        frequent ones, picked by the in-job word2vec). Saved in the checkpoint,
        so inference reproduces the raw_id -> compact_index mapping with no
        external artifact (just ``searchsorted``).
      - ``emb``: an ``(K+1, emb_dim)`` table; row 0 = OOV / padding zero vector,
        rows ``1..K`` are word2vec-initialized and then fine-tuned end to end.

    Forward maps each raw id to its compact index (exact match in ``kept_ids``,
    else OOV=0) and looks up the embedding.
    """

    def __init__(self, kept_size: int, emb_dim: int, freeze: bool = False) -> None:
        super().__init__()
        self.kept_size = int(kept_size)
        self.emb_dim = int(emb_dim)
        self.freeze = bool(freeze)
        self.register_buffer(
            'kept_ids', torch.zeros(max(self.kept_size, 1), dtype=torch.long))
        self.emb = nn.Embedding(self.kept_size + 1, emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.emb.weight.data)
        self.emb.weight.data[0].zero_()
        if self.freeze:
            # Use the word2vec vectors as-is: no gradient, never fine-tuned.
            self.emb.weight.requires_grad_(False)

    @torch.no_grad()
    def set_init(self, kept_ids_arr, init_matrix) -> None:
        """Fill ``kept_ids`` and ``emb`` from word2vec results (train-time only)."""
        kept = torch.as_tensor(kept_ids_arr, dtype=torch.long)
        assert kept.numel() == self.kept_size, (kept.numel(), self.kept_size)
        if self.kept_size > 0:
            self.kept_ids.copy_(kept)
        self.emb.weight.data.copy_(
            torch.as_tensor(init_matrix, dtype=self.emb.weight.dtype))

    def forward(self, raw_ids: torch.Tensor) -> torch.Tensor:
        """raw_ids: (B, L) long -> (B, L, emb_dim)."""
        if self.kept_size == 0:
            return raw_ids.new_zeros(
                (*raw_ids.shape, self.emb_dim), dtype=self.emb.weight.dtype)
        flat = raw_ids.reshape(-1).long()
        pos = torch.searchsorted(self.kept_ids, flat)
        pos = pos.clamp(max=self.kept_size - 1)
        matched = (self.kept_ids[pos] == flat) & (flat > 0)
        idx = torch.where(matched, pos + 1, torch.zeros_like(flat))
        return self.emb(idx.view_as(raw_ids))


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, Pool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]

    ``Pool`` is selected by ``query_pool_mode``:
      - 'mean': masked mean pooling over valid behaviors (v24 behavior,
        target-agnostic).
      - 'target_attn': DIN-style target-aware weighted pooling (ported from
        v17). Each behavior's relevance to the candidate item is scored by a
        per-sequence MLP over [s, t, s-t, s*t], softmax-normalized, then used
        as pooling weights, so every Q is seeded with a candidate-aware
        summary. Requires ``target_item`` in ``forward``.
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        query_pool_mode: str = 'mean',
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.query_pool_mode = query_pool_mode

        # DIN-style activation unit per sequence for target-aware pooling.
        # Mirrors DINTargetAttention scoring: MLP over the 4-way interaction
        # [seq, target, seq-target, seq*target] -> scalar relevance score.
        if query_pool_mode == 'target_attn':
            self.pool_act_units = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(4 * d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 1),
                )
                for _ in range(num_sequences)
            ])

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        target_item: Optional[torch.Tensor] = None,
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.
            target_item: (B, D), candidate-item vector. Required when
                ``query_pool_mode == 'target_attn'``; ignored otherwise.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)
        use_target_attn = (
            self.query_pool_mode == 'target_attn' and target_item is not None)

        q_tokens_list = []
        for i in range(self.num_sequences):
            seq_i = seq_tokens_list[i]
            pad_mask = seq_padding_masks[i]  # (B, L_i), True = padding

            if use_target_attn:
                # DIN-style target-aware weighted pooling.
                Bi, Li, Di = seq_i.shape
                t_exp = target_item.unsqueeze(1).expand(Bi, Li, Di)  # (B, L, D)
                din_feat = torch.cat(
                    [seq_i, t_exp, seq_i - t_exp, seq_i * t_exp], dim=-1)  # (B, L, 4D)
                scores = self.pool_act_units[i](din_feat).squeeze(-1)  # (B, L)
                scores = scores.masked_fill(pad_mask, float('-inf'))
                weights = torch.softmax(scores, dim=-1)
                # All-padding rows -> all -inf -> nan; zero them so the pooled
                # vector is 0 (matches the mean-pool clamp behavior).
                weights = torch.nan_to_num(weights, nan=0.0)
                seq_pooled = torch.bmm(weights.unsqueeze(1), seq_i).squeeze(1)  # (B, D)
            else:
                # MeanPool(Seq_i): masked mean over valid behaviors.
                valid_mask_expanded = (~pad_mask).unsqueeze(-1).float()  # (B, L, 1)
                seq_sum = (seq_i * valid_mask_expanded).sum(dim=1)  # (B, D)
                seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
                seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full'
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        hash_bucket_size: int = 0,
        hash_gate_init: float = -0.75,
        w2v_slots: Optional[dict] = None,
        w2v_freeze: bool = False,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
            hash_bucket_size: If >0, features skipped due to high vocab get a
                hash embedding of this many buckets instead of a zero vector.
            hash_gate_init: Initial gate logit for hash embeddings.
            w2v_slots: Optional ``{fid_idx: kept_size}``; these feature indices
                use compressed word2vec embeddings instead of normal/hash ones.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.hash_bucket_size = hash_bucket_size
        self.w2v_slots = {int(k): int(v) for k, v in (w2v_slots or {}).items()}
        self.w2v_freeze = bool(w2v_freeze)
        self.w2v_embs = nn.ModuleDict({
            str(fid_idx): W2VSeqEmbedding(kept_size, emb_dim, freeze=self.w2v_freeze)
            for fid_idx, kept_size in self.w2v_slots.items()
        })

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        hash_embs = []
        self._hash_index = []
        for fid_idx, (vs, offset, length) in enumerate(feature_specs):
            high_card = emb_skip_threshold > 0 and int(vs) > emb_skip_threshold
            use_w2v = fid_idx in self.w2v_slots
            skip = use_w2v or int(vs) <= 0 or high_card
            if skip:
                embs.append(None)
                if high_card and hash_bucket_size > 0 and not use_w2v:
                    self._hash_index.append(len(hash_embs))
                    he = nn.Embedding(hash_bucket_size + 1, emb_dim, padding_idx=0)
                    nn.init.xavier_normal_(he.weight.data)
                    he.weight.data[0, :] = 0
                    hash_embs.append(he)
                else:
                    self._hash_index.append(-1)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
                self._hash_index.append(-1)
        self.embs = nn.ModuleList([e for e in embs if e is not None])

        if hash_embs:
            self.hash_embs = nn.ModuleList(hash_embs)
            self.hash_gate = nn.Parameter(torch.full((1,), hash_gate_init))
            logging.info(
                "RankMixerNSTokenizer: %d high-card fields recovered via hash "
                "(bucket=%d, gate_init=%.2f)",
                len(hash_embs), hash_bucket_size, hash_gate_init,
            )
        else:
            self.hash_embs = None
            self.hash_gate = None
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )
        if self.w2v_slots:
            logging.info("RankMixerNSTokenizer word2vec slots: %s (freeze=%s)",
                         self.w2v_slots, self.w2v_freeze)

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if str(fid_idx) in self.w2v_embs:
                    w2v = self.w2v_embs[str(fid_idx)]
                    if length == 1:
                        fid_emb = w2v(int_feats[:, offset:offset + 1].long()).squeeze(1)
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = w2v(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                elif emb_real_idx == -1:
                    hash_idx = self._hash_index[fid_idx]
                    if hash_idx >= 0 and self.hash_embs is not None:
                        gate = torch.sigmoid(self.hash_gate)
                        if length == 1:
                            raw = int_feats[:, offset].long()
                            hashed = torch.where(
                                raw > 0,
                                torch.remainder(raw, self.hash_bucket_size) + 1,
                                raw.new_zeros(()),
                            )
                            fid_emb = self.hash_embs[hash_idx](hashed) * gate
                        else:
                            vals = int_feats[:, offset:offset + length].long()
                            hashed = torch.where(
                                vals > 0,
                                torch.remainder(vals, self.hash_bucket_size) + 1,
                                vals.new_zeros(()),
                            )
                            emb_all = self.hash_embs[hash_idx](hashed)
                            mask = (vals != 0).float().unsqueeze(-1)
                            count = mask.sum(dim=1).clamp(min=1)
                            fid_emb = (emb_all * mask).sum(dim=1) / count
                            fid_emb = fid_emb * gate
                    else:
                        fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)

    def load_w2v_init(self, init_data: dict, side: str) -> None:
        """Fill NS word2vec embeddings for this tokenizer."""
        for (init_side, fid_idx), (kept_ids, mat) in init_data.items():
            if init_side != side:
                continue
            self.w2v_embs[str(int(fid_idx))].set_init(kept_ids, mat)


class _DenseBranchProjector(nn.Module):
    """Project one value-scale-homogeneous dense branch to one d_model token.

    Applies a per-branch value transform (raw / log1p_nonneg / signed_log1p)
    then a small MLP. No presence channel and no gate: branch isolation plus the
    downstream reducer already give each value-scale its own learned projection
    and let weak/noisy branches be attenuated data-drivenly.
    """

    def __init__(self, in_dim: int, d_model: int, transform: str,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("_DenseBranchProjector requires in_dim > 0")
        self.transform = transform
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def _xform(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform == 'raw':
            return x
        if self.transform == 'log1p_nonneg':
            return torch.log1p(torch.clamp(x, min=0.0))
        if self.transform == 'signed_log1p':
            return torch.sign(x) * torch.log1p(torch.abs(x))
        raise ValueError(f"Unknown dense transform: {self.transform}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self._xform(x))


class _BranchedDenseProjector(nn.Module):
    """Generic scale-aware dense projector: split fids into value-scale branches,
    project each to a token, then aggregate all branch tokens via one MLP into
    ``out_tokens`` external tokens. Aggregation keeps noisy/weak branches from
    being handed to the backbone directly (they get attenuated inside this
    module instead).
    """

    def __init__(self, feature_specs, d_model: int, branch_defs,
                 out_tokens: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.out_tokens = out_tokens
        self.fid_to_slice = {int(fid): (int(off), int(ln)) for fid, off, ln in feature_specs}

        covered = set()
        self.branch_specs = []  # (name, fids, transform)
        self.branch_projectors = nn.ModuleDict()
        for name, fids, transform in branch_defs:
            existing = [f for f in fids if f in self.fid_to_slice]
            if not existing:
                continue
            dim = sum(self.fid_to_slice[f][1] for f in existing)
            self.branch_specs.append((name, existing, transform))
            self.branch_projectors[name] = _DenseBranchProjector(
                in_dim=dim, d_model=d_model, transform=transform, dropout=dropout)
            covered.update(existing)

        other = [f for f, _, _ in self.fid_to_slice_items() if f not in covered]
        if other:
            dim = sum(self.fid_to_slice[f][1] for f in other)
            self.branch_specs.append(('dense_other', other, 'raw'))
            self.branch_projectors['dense_other'] = _DenseBranchProjector(
                in_dim=dim, d_model=d_model, transform='raw', dropout=dropout)

        if not self.branch_specs:
            raise ValueError("_BranchedDenseProjector has no active branches")

        n = len(self.branch_specs)
        self.reducer = nn.Sequential(
            nn.Linear(n * d_model, out_tokens * d_model),
            nn.LayerNorm(out_tokens * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)
        logging.info(
            f"_BranchedDenseProjector: {n} branches -> {out_tokens} token(s); "
            f"branches={[s[0] for s in self.branch_specs]}")

    def fid_to_slice_items(self):
        # preserve insertion order of feature_specs
        return [(f, sl[0], sl[1]) for f, sl in self.fid_to_slice.items()]

    def _gather(self, x, fids):
        parts = []
        for f in fids:
            off, ln = self.fid_to_slice[f]
            parts.append(x[:, off:off + ln])
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        toks = []
        for name, fids, _t in self.branch_specs:
            toks.append(self.branch_projectors[name](self._gather(x, fids)))
        flat = torch.cat(toks, dim=-1)
        out = self.reducer(flat).view(x.shape[0], self.out_tokens, self.d_model)
        return self.out_norm(out)  # (B, out_tokens, d_model)


# user_dense: 8 value-scale branches -> 2 external tokens. No presence: EDA shows
# user_dense fids are essentially fully populated (62-66/118/121 have 0% missing,
# 130/131 ~1%, 87's 61.9% zeros are a sparse embedding's true zeros not missing),
# so a presence channel would be a near-constant noise column.
_USER_DENSE_BRANCH_DEFS = [
    ('ud_fid61', [61], 'raw'),
    ('ud_fid87', [87], 'raw'),
    ('ud_emb_89_90_91', [89, 90, 91], 'raw'),
    ('ud_fid120', [120], 'raw'),
    ('ud_heavy_count_62_66', [62, 63, 64, 65, 66], 'log1p_nonneg'),
    ('ud_mid_count_118_121_123', [118, 121, 123], 'log1p_nonneg'),
    ('ud_sparse_score_130_131', [130, 131], 'signed_log1p'),
    ('ud_wide_score_132', [132], 'raw'),
]

# item_dense: value-scale branches -> 2 external tokens, mirroring the user
# dense path so original item dense and synthetic history branches have more
# capacity before entering the NS stream.
# fid129 has an extreme tail (p99=28 vs max=6599) -> signed_log1p.
# fids 10001-10012 are synthetic item history features already log/ratio/entropy
# scaled in dataset.py, so they stay on the raw branch.
_ITEM_DENSE_BRANCH_DEFS = [
    ('id_emb_127_128', [127, 128], 'raw'),
    ('id_score_124', [124], 'raw'),
    ('id_heavy_129', [129], 'signed_log1p'),
    ('id_hist_10001_10012', list(range(10001, 10013)), 'raw'),
]


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    @staticmethod
    def _append_calendar_int_specs(
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        enabled: bool,
    ) -> Tuple[List[Tuple[int, int, int]], List[List[int]]]:
        """Append sample-level calendar ids to the int feature stream."""
        feature_specs = list(feature_specs)
        groups = [list(group) for group in groups]
        if not enabled:
            return feature_specs, groups

        next_offset = 0
        for _, offset, length in feature_specs:
            next_offset = max(next_offset, offset + length)

        calendar_indices = []
        for vocab_size in USER_CALENDAR_INT_VOCABS:
            calendar_indices.append(len(feature_specs))
            feature_specs.append((vocab_size, next_offset, 1))
            next_offset += 1

        if groups:
            groups[-1].extend(calendar_indices)
        else:
            groups.append(calendar_indices)
        return feature_specs, groups

    @staticmethod
    def _append_calendar_dense_specs(
        feature_specs: Optional[List[Tuple[int, int, int]]],
        dense_dim: int,
        enabled: bool,
    ) -> Optional[List[Tuple[int, int, int]]]:
        """Append sample-level calendar dense features to user dense specs."""
        if not enabled:
            return feature_specs
        specs = list(feature_specs) if feature_specs is not None else [(0, 0, dense_dim)]
        next_offset = 0
        for _, offset, length in specs:
            next_offset = max(next_offset, offset + length)
        for i in range(USER_CALENDAR_DENSE_DIM):
            specs.append((USER_CALENDAR_DENSE_FID_BASE + i, next_offset + i, 1))
        return specs

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        use_calendar_time_features: bool = False,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        seq_hash_bucket_size: int = 0,
        seq_hash_gate_init: float = -0.75,
        ns_hash_bucket_size: int = 0,
        ns_hash_gate_init: float = -0.75,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        user_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        item_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        # Supervised contrastive auxiliary head (v22).
        use_supcon: bool = False,
        supcon_proj_dim: int = 0,
        # Query pooling for MultiSeqQueryGenerator (v25: DIN-MLP target-aware
        # weighted pooling, ported from v17).
        query_pool_mode: str = 'mean',
        # word2vec warm-started compressed-vocab embeddings (v28).
        # {domain: {slot_idx: kept_size}} — these seq slots use a W2VSeqEmbedding
        # instead of the normal/hash embedding.
        seq_w2v_slots: Optional[dict] = None,
        # NS-side word2vec warm-started compressed-vocab embeddings.
        # {"user": {fid_idx: kept_size}, "item": {fid_idx: kept_size}}.
        ns_w2v_slots: Optional[dict] = None,
        # v29: freeze the word2vec embeddings (use them as-is, no fine-tuning).
        seq_w2v_freeze: bool = False,
        ns_w2v_freeze: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.use_calendar_time_features = use_calendar_time_features
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.seq_hash_bucket_size = seq_hash_bucket_size
        self.ns_tokenizer_type = ns_tokenizer_type
        self.ns_w2v_slots = {
            side: {int(k): int(v) for k, v in slotmap.items()}
            for side, slotmap in (ns_w2v_slots or {}).items()
        }
        self.ns_w2v_freeze = bool(ns_w2v_freeze)
        self.sample_calendar_int_dim = (
            len(USER_CALENDAR_INT_VOCABS) if use_calendar_time_features else 0)
        self.sample_calendar_dense_dim = (
            USER_CALENDAR_DENSE_DIM if use_calendar_time_features else 0)

        user_int_feature_specs, user_ns_groups = self._append_calendar_int_specs(
            user_int_feature_specs, user_ns_groups, use_calendar_time_features)
        item_int_feature_specs, item_ns_groups = self._append_calendar_int_specs(
            item_int_feature_specs, item_ns_groups, use_calendar_time_features)
        user_dense_feature_specs = self._append_calendar_dense_specs(
            user_dense_feature_specs, user_dense_dim, use_calendar_time_features)

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=ns_hash_bucket_size,
                hash_gate_init=ns_hash_gate_init,
                w2v_slots=self.ns_w2v_slots.get('user', {}),
                w2v_freeze=self.ns_w2v_freeze,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=ns_hash_bucket_size,
                hash_gate_init=ns_hash_gate_init,
                w2v_slots=self.ns_w2v_slots.get('item', {}),
                w2v_freeze=self.ns_w2v_freeze,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection: 8 value-scale branches -> 2 tokens,
        # no presence/gate (EDA: user_dense is essentially fully populated).
        self.has_user_dense = user_dense_dim > 0
        self.num_user_dense_tokens = 2 if self.has_user_dense else 0
        if self.has_user_dense:
            if user_dense_feature_specs is None:
                user_dense_feature_specs = [(0, 0, user_dense_dim)]
            self.user_dense_proj = _BranchedDenseProjector(
                feature_specs=user_dense_feature_specs,
                d_model=d_model,
                branch_defs=_USER_DENSE_BRANCH_DEFS,
                out_tokens=2,
                dropout=dropout_rate,
            )

        # Item dense feature projection: value-scale branches -> 2 tokens,
        # no presence. fid129's extreme tail handled via signed_log1p inside
        # its branch.
        self.has_item_dense = item_dense_dim > 0
        self.num_item_dense_tokens = 2 if self.has_item_dense else 0
        if self.has_item_dense:
            if item_dense_feature_specs is None:
                item_dense_feature_specs = [(0, 0, item_dense_dim)]
            self.item_dense_proj = _BranchedDenseProjector(
                feature_specs=item_dense_feature_specs,
                d_model=d_model,
                branch_defs=_ITEM_DENSE_BRANCH_DEFS,
                out_tokens=2,
                dropout=dropout_rate,
            )

        # Candidate-item target token (v25): rebuilt from item NS tokens (+ item
        # dense token when present), used by target-aware query pooling. Only
        # created when that pooling mode is active so 'mean' stays equivalent
        # to v24.
        self.query_pool_mode = query_pool_mode
        if self.query_pool_mode == 'target_attn':
            self.item_target_num_tokens = num_item_ns + self.num_item_dense_tokens
            self.item_target_proj = nn.Sequential(
                nn.Linear(self.item_target_num_tokens * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )

        # Total NS token count
        self.num_ns = (num_user_ns + self.num_user_dense_tokens
                       + num_item_ns + self.num_item_dense_tokens)

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        # word2vec warm-started slots (v28): {domain: {slot_idx: kept_size}}.
        # Normalize keys (may arrive as strings from train_config at inference).
        self.seq_w2v_slots = {}
        for dom, slotmap in (seq_w2v_slots or {}).items():
            self.seq_w2v_slots[dom] = {int(k): int(v) for k, v in slotmap.items()}
        self.seq_w2v_freeze = bool(seq_w2v_freeze)
        self._seq_w2v = nn.ModuleDict()  # domain -> ModuleDict{str(slot): W2VSeqEmbedding}
        for dom, slotmap in self.seq_w2v_slots.items():
            self._seq_w2v[dom] = nn.ModuleDict({
                str(slot): W2VSeqEmbedding(K, emb_dim, freeze=self.seq_w2v_freeze)
                for slot, K in slotmap.items()})
        if self.seq_w2v_slots:
            logging.info("seq word2vec embeddings: %s (freeze=%s)",
                         self.seq_w2v_slots, self.seq_w2v_freeze)

        def _make_seq_embs(vocab_sizes, w2v_set):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold, with no vocab info (vs<=0), or handled by a
            word2vec embedding (w2v_set)."""
            embs_raw = []
            for i, vs in enumerate(vocab_sizes):
                skip = (i in w2v_set) or int(vs) <= 0 or (
                    emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            w2v_set = set(self.seq_w2v_slots.get(domain, {}).keys())
            embs, idx_map, is_id = _make_seq_embs(vs, w2v_set)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Hash Embeddings for skipped seq fields ==================
        self._seq_hash_embs = nn.ModuleDict()
        self._seq_hash_gates = nn.ParameterDict()
        self._seq_hash_index = {}

        if seq_hash_bucket_size > 0:
            for domain in self.seq_domains:
                vs_list = seq_vocab_sizes[domain]
                idx_map = self._seq_emb_index[domain]
                w2v_set = set(self.seq_w2v_slots.get(domain, {}).keys())
                hash_embs = []
                hash_index = []

                for i, vs_i in enumerate(vs_list):
                    if idx_map[i] == -1 and int(vs_i) > 0 and i not in w2v_set:
                        hash_index.append(len(hash_embs))
                        emb = nn.Embedding(
                            seq_hash_bucket_size + 1, emb_dim, padding_idx=0)
                        nn.init.xavier_normal_(emb.weight.data)
                        emb.weight.data[0, :] = 0
                        hash_embs.append(emb)
                    else:
                        hash_index.append(-1)

                self._seq_hash_index[domain] = hash_index
                if hash_embs:
                    self._seq_hash_embs[domain] = nn.ModuleList(hash_embs)
                    self._seq_hash_gates[domain] = nn.Parameter(
                        torch.full((1,), seq_hash_gate_init))
                    logging.info(
                        "seq_hash: %s - %d/%d skipped fields recovered "
                        "(bucket=%d, gate_init=%.2f)",
                        domain, len(hash_embs), len(vs_list),
                        seq_hash_bucket_size, seq_hash_gate_init,
                    )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== Calendar Time Embeddings (optional) ==================
        if self.use_calendar_time_features:
            self.seq_hour_embedding = nn.Embedding(
                NUM_CALENDAR_HOUR_IDS, d_model, padding_idx=0)
            self.seq_weekday_embedding = nn.Embedding(
                NUM_CALENDAR_WEEKDAY_IDS, d_model, padding_idx=0)
            self.seq_period_embedding = nn.Embedding(
                NUM_CALENDAR_PERIOD_IDS, d_model, padding_idx=0)
            self.seq_calendar_time_gate = nn.Parameter(torch.zeros(1))

            # Day-type embeddings: is_weekend, is_holiday (vocab=3: pad/no/yes)
            self.seq_weekend_embedding = nn.Embedding(3, d_model, padding_idx=0)
            self.seq_holiday_embedding = nn.Embedding(3, d_model, padding_idx=0)
            self.seq_day_type_gate = nn.Parameter(torch.zeros(1))

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
            query_pool_mode=query_pool_mode,
        )
        if query_pool_mode == 'target_attn':
            logging.info(
                "Query pooling: DIN-MLP target-aware weighted pooling "
                "(per-sequence activation unit over [s, t, s-t, s*t])")

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # ── SupCon projection head (v22) ──
        # Maps the final pre-classifier representation (which already fuses
        # user / item / sequence) to a contrastive embedding. The SupCon loss
        # (computed in the trainer) pulls same-label samples together.
        self.use_supcon = bool(use_supcon)
        if self.use_supcon:
            sc_dim = supcon_proj_dim if supcon_proj_dim > 0 else d_model
            self.supcon_proj = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, sc_dim))
            logging.info("SupCon head enabled: proj_dim=%s (on final output)", sc_dim)

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
            if getattr(tokenizer, 'hash_embs', None) is not None:
                for emb in tokenizer.hash_embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

        if self.use_calendar_time_features:
            for emb in [
                self.seq_hour_embedding,
                self.seq_weekday_embedding,
                self.seq_period_embedding,
            ]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1
        if self.use_calendar_time_features:
            skip_count += 3

        for domain in self.seq_domains:
            if domain in self._seq_hash_embs:
                for emb in self._seq_hash_embs[domain]:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            if getattr(tokenizer, 'hash_embs', None) is not None:
                for emb in tokenizer.hash_embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        # requires_grad filter excludes frozen embeddings (e.g. frozen word2vec
        # tables) so they are not handed to the optimizer at all.
        return [p for p in self.parameters()
                if p.requires_grad and p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters()
                if p.requires_grad and p.data_ptr() not in sparse_ptrs]

    def _calendar_time_embedding(
        self,
        time_feats: torch.Tensor,
        hour_embedding: nn.Embedding,
        weekday_embedding: nn.Embedding,
        period_embedding: nn.Embedding,
    ) -> torch.Tensor:
        """Embed [hour_id, weekday_id, period_id] categorical time features."""
        hour_ids = time_feats[..., 0].long()
        weekday_ids = time_feats[..., 1].long()
        period_ids = time_feats[..., 2].long()
        return (
            hour_embedding(hour_ids)
            + weekday_embedding(weekday_ids)
            + period_embedding(period_ids)
        )

    def _with_sample_calendar_int(
        self,
        int_feats: torch.Tensor,
        time_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Append sample-level calendar ids to an int feature tensor."""
        if self.sample_calendar_int_dim == 0:
            return int_feats
        return torch.cat([int_feats, time_feats.long()], dim=-1)

    def _calendar_dense_features(self, time_feats: torch.Tensor) -> torch.Tensor:
        """Build bounded continuous calendar features for the user-dense path."""
        B = time_feats.shape[0]
        if self.sample_calendar_dense_dim == 0:
            return time_feats.new_zeros(B, 0, dtype=torch.float32)

        time_float = time_feats.to(dtype=torch.float32)
        valid = (time_feats[..., 0] > 0).to(dtype=torch.float32).unsqueeze(-1)
        hour = (time_float[..., 0] - 1.0).clamp(min=0.0, max=23.0)
        weekday = (time_float[..., 1] - 1.0).clamp(min=0.0, max=6.0)
        hour_angle = hour * (2.0 * math.pi / 24.0)
        weekday_angle = weekday * (2.0 * math.pi / 7.0)
        dense_time = torch.stack([
            torch.sin(hour_angle),
            torch.cos(hour_angle),
            torch.sin(weekday_angle),
            torch.cos(weekday_angle),
        ], dim=-1)
        return dense_time * valid

    def _with_sample_calendar_dense(
        self,
        dense_feats: torch.Tensor,
        time_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Append sample-level dense time features to user dense features."""
        if self.sample_calendar_dense_dim == 0:
            return dense_feats
        dense_time = self._calendar_dense_features(time_feats).to(dtype=dense_feats.dtype)
        return torch.cat([dense_feats, dense_time], dim=-1)

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        calendar_time_feats: Optional[torch.Tensor] = None,
        day_type_feats: Optional[torch.Tensor] = None,
        domain: Optional[str] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        hash_index = self._seq_hash_index.get(domain) if domain else None
        hash_embs = (
            self._seq_hash_embs[domain]
            if domain and domain in self._seq_hash_embs else None
        )
        hash_gate = (
            torch.sigmoid(self._seq_hash_gates[domain])
            if domain and domain in self._seq_hash_gates else None
        )
        w2v_embs = self._seq_w2v[domain] if (domain and domain in self._seq_w2v) else None

        emb_list = []
        for i in range(S):
            # word2vec warm-started compressed-vocab embedding (v28) takes
            # precedence over the normal / hash path for this slot.
            if w2v_embs is not None and str(i) in w2v_embs:
                e = w2v_embs[str(i)](seq[:, i, :])
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
                continue
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                h_idx = hash_index[i] if (hash_index and i < len(hash_index)) else -1
                if h_idx >= 0 and hash_embs is not None and hash_gate is not None:
                    raw_ids = seq[:, i, :]
                    hashed = torch.where(
                        raw_ids > 0,
                        torch.remainder(raw_ids, self.seq_hash_bucket_size) + 1,
                        raw_ids.new_zeros((), dtype=raw_ids.dtype),
                    )
                    e = hash_embs[h_idx](hashed) * hash_gate
                    if is_id[i] and self.training:
                        e = self.seq_id_emb_dropout(e)
                    emb_list.append(e)
                else:
                    emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        if self.use_calendar_time_features and calendar_time_feats is not None:
            calendar_time_emb = self._calendar_time_embedding(
                calendar_time_feats,
                self.seq_hour_embedding,
                self.seq_weekday_embedding,
                self.seq_period_embedding,
            ).to(dtype=token_emb.dtype)
            gate = torch.tanh(self.seq_calendar_time_gate).to(dtype=token_emb.dtype)
            token_emb = token_emb + gate * calendar_time_emb

        if self.use_calendar_time_features and day_type_feats is not None:
            weekend_emb = self.seq_weekend_embedding(day_type_feats[..., 0].long())
            holiday_emb = self.seq_holiday_embedding(day_type_feats[..., 1].long())
            dt_emb = (weekend_emb + holiday_emb).to(dtype=token_emb.dtype)
            dt_gate = torch.tanh(self.seq_day_type_gate).to(dtype=token_emb.dtype)
            token_emb = token_emb + dt_gate * dt_emb

        return token_emb

    def load_seq_w2v_init(self, init_data: dict) -> None:
        """Fill the word2vec warm-start embeddings (train-time only).

        Args:
            init_data: ``{(domain, slot_idx): (kept_ids, init_matrix)}``.
        """
        for (domain, slot), (kept_ids, mat) in init_data.items():
            mod = self._seq_w2v[domain][str(int(slot))]
            mod.set_init(kept_ids, mat)
        logging.info("Loaded word2vec init for %d seq id feature(s)", len(init_data))

    def load_ns_w2v_init(self, init_data: dict) -> None:
        """Fill NS-side word2vec warm-start embeddings (train-time only).

        Args:
            init_data: ``{(side, fid_idx): (kept_ids, init_matrix)}``, where
                ``side`` is ``"user"`` or ``"item"``.
        """
        if hasattr(self.user_ns_tokenizer, 'load_w2v_init'):
            self.user_ns_tokenizer.load_w2v_init(init_data, 'user')
        if hasattr(self.item_ns_tokenizer, 'load_w2v_init'):
            self.item_ns_tokenizer.load_w2v_init(init_data, 'item')
        logging.info("Loaded NS word2vec init for %d feature(s)", len(init_data))

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def _build_target_item(
        self,
        item_ns: torch.Tensor,
        item_dense_tok: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Builds the candidate-item target vector for target-aware pooling.

        Returns ``None`` unless ``query_pool_mode == 'target_attn'``, in which
        case the item NS tokens (plus the item dense token when present) are
        flattened and projected to a single (B, D) target vector.
        """
        if self.query_pool_mode != 'target_attn':
            return None
        parts = [item_ns]
        if item_dense_tok is not None:
            parts.append(item_dense_tok)
        B = item_ns.shape[0]
        return self.item_target_proj(torch.cat(parts, dim=1).reshape(B, -1))

    def forward(self, inputs: ModelInput, return_repr: bool = False):
        """Runs the forward pass of the PCVRHyFormer model.

        When ``return_repr`` is True (v22 SupCon training), also returns the
        projected contrastive embedding of the final representation as
        ``(logits, z)``.
        """
        # 1. NS tokens: grouped projection
        user_int_feats = self._with_sample_calendar_int(
            inputs.user_int_feats, inputs.time_feats)
        item_int_feats = self._with_sample_calendar_int(
            inputs.item_int_feats, inputs.time_feats)
        user_ns = self.user_ns_tokenizer(user_int_feats)   # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(item_int_feats)   # (B, num_item_groups, D)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_feats = self._with_sample_calendar_dense(
                inputs.user_dense_feats, inputs.time_feats)
            user_dense_tok = self.user_dense_proj(user_dense_feats)  # (B, 2, D)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        item_dense_tok = None
        if self.has_item_dense:
            item_dense_tok = self.item_dense_proj(inputs.item_dense_feats)  # (B, 2, D)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)
        target_item = self._build_target_item(item_ns, item_dense_tok)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_time_feats.get(domain),
                day_type_feats=inputs.seq_day_type_feats.get(domain),
                domain=domain)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list, target_item=target_item)

        # 4. Dropout + MultiSeqHyFormerBlock stack + output projection
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training
        )

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        if return_repr and self.use_supcon:
            return logits, self.supcon_proj(output)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        user_int_feats = self._with_sample_calendar_int(
            inputs.user_int_feats, inputs.time_feats)
        item_int_feats = self._with_sample_calendar_int(
            inputs.item_int_feats, inputs.time_feats)
        user_ns = self.user_ns_tokenizer(user_int_feats)
        item_ns = self.item_ns_tokenizer(item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_feats = self._with_sample_calendar_dense(
                inputs.user_dense_feats, inputs.time_feats)
            user_dense_tok = self.user_dense_proj(user_dense_feats)  # (B, 2, D)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        item_dense_tok = None
        if self.has_item_dense:
            item_dense_tok = self.item_dense_proj(inputs.item_dense_feats)  # (B, 2, D)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)
        target_item = self._build_target_item(item_ns, item_dense_tok)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_time_feats.get(domain),
                day_type_feats=inputs.seq_day_type_feats.get(domain),
                domain=domain)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list, target_item=target_item)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False
        )

        logits = self.clsfier(output)
        return logits, output
