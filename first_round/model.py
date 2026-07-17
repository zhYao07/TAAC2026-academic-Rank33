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
    user_dense_presence_feats: torch.Tensor
    item_dense_presence_feats: torch.Tensor
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
USER_SUM_DENSE_RANGE = (0, 256)
USER_SIDE_DENSE_RANGE_A = (256, 568)
USER_ADS_DENSE_RANGE = (568, 888)
USER_SIDE_DENSE_RANGE_B = (888, 918)




def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Return the standard ALiBi per-head slopes."""

    def _power_of_two_slopes(n: int) -> List[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = _power_of_two_slopes(num_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(num_heads))
        slopes = _power_of_two_slopes(closest_power)
        extra = _get_alibi_slopes(2 * closest_power).tolist()
        slopes.extend(extra[0::2][:num_heads - closest_power])
    return torch.tensor(slopes, dtype=torch.float32)


class ALiBiPositionBias(nn.Module):
    """Build additive ALiBi attention bias for newest-first sequences."""

    def __init__(self, num_heads: int, bias_scale: float = 1.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.bias_scale = float(bias_scale)
        self.register_buffer(
            "slopes", _get_alibi_slopes(num_heads), persistent=False)

    def forward(
        self,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
        q_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_pos = torch.arange(k_len, device=device, dtype=torch.float32)
        if q_positions is None:
            q_pos = torch.arange(q_len, device=device, dtype=torch.float32)
            q_pos = q_pos.view(1, q_len, 1)
        else:
            q_pos = q_positions.to(device=device, dtype=torch.float32).unsqueeze(-1)
        distance = (q_pos - key_pos.view(1, 1, k_len)).abs()
        bias = -distance.unsqueeze(1) * self.slopes.view(
            1, self.num_heads, 1, 1).to(device=device)
        bias = bias * self.bias_scale
        return bias.to(dtype=dtype)

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
    cos_ = cos[:, :L, :].unsqueeze(1).to(dtype=x.dtype)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1).to(dtype=x.dtype)
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
        alibi_bias: Optional[torch.Tensor] = None,
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
            alibi_bias: Optional additive bias shaped (1 or B, num_heads, Lq, Lk).
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

        # 4. Convert masks/bias to SDPA format
        sdpa_attn_mask = None
        if alibi_bias is not None:
            sdpa_attn_mask = alibi_bias.to(device=Q.device, dtype=Q.dtype)
            if sdpa_attn_mask.shape[0] == 1 and B > 1:
                sdpa_attn_mask = sdpa_attn_mask.expand(B, -1, -1, -1).clone()
            if key_padding_mask is not None:
                pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                pad_mask = pad_mask.expand(B, self.num_heads, Lq, Lk)
                sdpa_attn_mask = sdpa_attn_mask.masked_fill(
                    pad_mask, float('-inf'))
            if attn_mask is not None:
                blocked = (attn_mask != 0).to(device=Q.device)
                blocked = blocked.unsqueeze(0).unsqueeze(0).expand(
                    B, self.num_heads, Lq, Lk)
                sdpa_attn_mask = sdpa_attn_mask.masked_fill(
                    blocked, float('-inf'))
        else:
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


class DINTargetAttention(nn.Module):
    """DIN-style target attention for sequence query decoding.

    The current item representation acts as the target. For every sequence
    position, the activation unit scores ``seq_token`` against ``target_item``
    using the standard DIN interaction features. The pooled interest vector is
    then fused back into the generated query tokens, so QueryGenerator remains
    an active part of the block.

    When ``num_time_buckets > 0``, a learnable per-bucket additive bias is
    added to the attention scores so the model can learn explicit recency
    preferences.  The bias is zero-initialised (step-0 identical to vanilla DIN).
    """

    def __init__(
        self,
        d_model: int,
        num_queries: int,
        num_time_buckets: int = 0,
        hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries

        self.activation_unit = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        # Learnable recency bias per time bucket (zero-init).
        if num_time_buckets > 0:
            self.time_score_bias = nn.Embedding(
                num_time_buckets, 1, padding_idx=0)
            nn.init.zeros_(self.time_score_bias.weight)
        else:
            self.time_score_bias = None

        self.interest_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model * num_queries),
            nn.LayerNorm(d_model * num_queries),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.query_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_tokens: torch.Tensor,
        target_item: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_padding_mask: Optional[torch.Tensor] = None,
        time_bucket_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = seq_tokens.shape
        target_expanded = target_item.unsqueeze(1).expand(B, L, D)
        din_features = torch.cat(
            [seq_tokens, target_expanded,
             seq_tokens - target_expanded, seq_tokens * target_expanded],
            dim=-1,
        )

        scores = self.activation_unit(din_features).squeeze(-1)

        # Additive recency bias from time buckets.
        if self.time_score_bias is not None and time_bucket_ids is not None:
            tb = time_bucket_ids
            if tb.shape[1] > L:
                tb = tb[:, :L]
            elif tb.shape[1] < L:
                tb = F.pad(tb, (0, L - tb.shape[1]))
            scores = scores + self.time_score_bias(
                tb.long()).squeeze(-1).to(dtype=scores.dtype)

        if seq_padding_mask is not None:
            scores = scores.masked_fill(seq_padding_mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        interest = torch.bmm(weights.unsqueeze(1), seq_tokens).squeeze(1)
        decoded = self.interest_proj(torch.cat([interest, target_item], dim=-1))
        decoded = decoded.view(B, self.num_queries, -1)

        gate_input = torch.cat(
            [query_tokens, decoded, query_tokens * decoded], dim=-1)
        gate = self.query_gate(gate_input)
        return self.out_norm(query_tokens + gate * decoded)


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


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        use_time_decay: bool = False,
        time_decay_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.use_time_decay = bool(use_time_decay)
        self.time_decay_strength = float(time_decay_strength)

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
        seq_time_buckets_list: Optional[list] = None,
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.
            seq_time_buckets_list: Optional list of (B, L_i) relative-time
                bucket ids. Used only when use_time_decay=True; otherwise the
                original mean-pooling path is preserved.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # Pool(Seq_i). Default is the original unweighted mean.
            # Optional time-decay pooling is a dormant experiment and is never
            # active unless --use_qgen_time_decay is explicitly set.
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).to(
                dtype=seq_tokens_list[i].dtype)  # (B, L_i, 1)

            use_decay = (
                self.use_time_decay
                and self.time_decay_strength > 0.0
                and seq_time_buckets_list is not None
                and i < len(seq_time_buckets_list)
                and seq_time_buckets_list[i] is not None
            )
            if use_decay:
                # Bucket id 1 is the freshest non-padding bucket. Larger ids
                # are older/coarser buckets, so this softly emphasizes recent
                # events without changing sequence token shapes.
                bucket_age = (seq_time_buckets_list[i].to(
                    device=seq_tokens_list[i].device,
                    dtype=seq_tokens_list[i].dtype) - 1.0).clamp(min=0.0)
                decay = torch.exp(-self.time_decay_strength * bucket_age).unsqueeze(-1)
                weights = decay * valid_mask_expanded
                seq_sum = (seq_tokens_list[i] * weights).sum(dim=1)
                seq_count = weights.sum(dim=1).clamp(min=1e-6)
            else:
                seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
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
        alibi_position_bias: Optional[ALiBiPositionBias] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.
            alibi_position_bias: Optional ALiBi bias builder.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        alibi_bias = None
        if alibi_position_bias is not None:
            L = x.shape[1]
            alibi_bias = alibi_position_bias(L, L, x.device, x.dtype)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            alibi_bias=alibi_bias,
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

        Assumes the upstream sequence layout is **newest-first**:
        position 0 is the most recent event, positions ``valid_len..L-1`` are
        right-padding. The latest top_k tokens are therefore the first
        ``top_k`` slots; the returned sequence is still newest-first and the
        resulting padding sits at the tail of the gathered block.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding (located at
                the tail when ``valid_len < top_k``).
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE (identical across the
                batch since every sample uses positions 0..top_k-1).
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample (right-padding convention).
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)

        # Newest-first layout: just take the first top_k positions. We still
        # build an offsets table so the resulting padding mask and RoPE
        # position indices are shaped (B, top_k) and play well with the rest
        # of the encoder.
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)

        # Caller invokes this branch only when L > top_k, so the slice is
        # always safe; .contiguous() keeps the downstream zero-fill writeable.
        top_k_tokens = x[:, :self.top_k, :].contiguous()

        # Padding ends up at the tail: positions >= actual_k are padded.
        new_padding_mask = offsets >= actual_k.unsqueeze(1)  # (B, top_k)

        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).to(
            dtype=top_k_tokens.dtype)

        # Q-side RoPE positions: 0..top_k-1 (same as KV positions for those
        # slots, so attention stays self-consistent with the encoder's RoPE).
        position_indices = offsets

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        alibi_position_bias: Optional[ALiBiPositionBias] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.
            alibi_position_bias: Optional ALiBi bias builder.

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
            alibi_bias = None
            if alibi_position_bias is not None:
                alibi_bias = alibi_position_bias(
                    q.shape[1], L, q.device, q.dtype,
                    q_positions=q_pos_indices)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
                alibi_bias=alibi_bias,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask. NOTE: the upstream sequence is newest-first
            # (position 0 = newest), so the standard upper-triangular mask
            # produced by ``generate_square_subsequent_mask`` makes each
            # token attend to OLDER tokens (positions with j >= i). If you
            # actually want "newer token cannot peek at older history" you
            # need to transpose the mask. Left as-is for backward
            # compatibility; default ``--seq_causal`` is False.
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            alibi_bias = None
            if alibi_position_bias is not None:
                alibi_bias = alibi_position_bias(L, L, x.device, x.dtype)
            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                alibi_bias=alibi_bias,
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
    DIN target-based Query Decoding, then all Q tokens and shared NS tokens
    are merged for joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: Union[str, List[str]] = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        num_time_buckets: int = 0
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Allow per-domain encoder type. A bare string broadcasts to every
        # sequence, while a list of strings (length == num_sequences, aligned
        # with PCVRHyFormer.seq_domains order) picks an encoder per domain so
        # we can, e.g., run LongerEncoder on seq_d only and keep TransformerEncoder
        # everywhere else.
        if isinstance(seq_encoder_type, str):
            encoder_types = [seq_encoder_type] * num_sequences
        else:
            encoder_types = list(seq_encoder_type)
            if len(encoder_types) != num_sequences:
                raise ValueError(
                    f"seq_encoder_type list length {len(encoder_types)} does not "
                    f"match num_sequences {num_sequences}")

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=encoder_types[i],
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for i in range(num_sequences)
        ])

        # Independent DIN target-attention decoder per sequence.
        self.din_attns = nn.ModuleList([
            DINTargetAttention(
                d_model=d_model,
                num_queries=num_queries,
                num_time_buckets=num_time_buckets,
                hidden_mult=hidden_mult,
                dropout=dropout,
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
        target_item: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        seq_time_buckets_list: Optional[list] = None,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
        alibi_position_bias: Optional[ALiBiPositionBias] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            target_item: (B, D), current item target representation for DIN.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.
            alibi_position_bias: Optional ALiBi bias builder shared by all domains.

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
                alibi_position_bias=alibi_position_bias,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent DIN target-based Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            decoded_q_i = self.din_attns[i](
                query_tokens=q_tokens_list[i],
                target_item=target_item,
                seq_tokens=next_seqs[i],
                seq_padding_mask=next_masks[i],
                time_bucket_ids=(
                    seq_time_buckets_list[i]
                    if seq_time_buckets_list is not None else None),
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
                    fid_emb = int_feats.new_zeros(
                        int_feats.shape[0], self.emb_dim, dtype=torch.float32)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).to(dtype=emb_all.dtype).unsqueeze(-1)  # (B, length, 1)
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
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
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
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(
                        int_feats.shape[0], self.emb_dim, dtype=torch.float32)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).to(dtype=emb_all.dtype).unsqueeze(-1)
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

    def forward_with_field_embs(
        self, int_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Like forward() but also returns per-field embeddings for FAFE.

        Returns:
            ns_tokens: (B, num_ns_tokens, d_model)
            emb_list: list of per-field embeddings, each (B, 1, emb_dim)
            valid_mask: bool (num_fields,), False for skipped fields
        """
        all_embs = []
        valid_flags = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(
                        int_feats.shape[0], self.emb_dim, dtype=torch.float32)
                    valid_flags.append(False)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).to(dtype=emb_all.dtype).unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                    valid_flags.append(True)
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))

        chunks = cat_emb.split(self.chunk_dim, dim=-1)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))

        ns_tokens = torch.cat(tokens, dim=1)
        emb_list_3d = [e.unsqueeze(1) for e in all_embs]
        valid_mask = torch.tensor(valid_flags, dtype=torch.bool)
        return ns_tokens, emb_list_3d, valid_mask


class FieldGateFAFE(nn.Module):
    """Lightweight field-gate feature enhancement for sequence tokenization.

    It enhances one sequence-domain token without changing sequence length or
    the number of NS tokens. For each time step, every raw field embedding gets
    a field-specific projection and a learned gate. The weighted sum is added
    back to the original concat+Linear token as a residual.
    """

    def __init__(
        self,
        num_fields: int,
        emb_dim: int,
        d_model: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.d_model = d_model

        self.field_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_fields)
        ])
        self.field_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, max(8, emb_dim // 4)),
                nn.SiLU(),
                nn.Linear(max(8, emb_dim // 4), 1),
            )
            for _ in range(num_fields)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_dropout = nn.Dropout(dropout)
        # Start as a small residual branch to avoid destroying the existing
        # best model's sequence merge at the beginning of training.
        self.res_alpha = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        emb_list: List[torch.Tensor],
        base_token: torch.Tensor,
        valid_field_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Enhance the base token.

        Args:
            emb_list: list of F tensors, each [B, L, emb_dim].
            base_token: original concat+Linear token, [B, L, d_model].
            valid_field_mask: optional bool tensor [F], False masks a skipped
                field out of the softmax gate.
        """
        proj_parts = []
        gate_scores = []
        for i, emb in enumerate(emb_list):
            proj_parts.append(self.field_projs[i](emb).unsqueeze(2))       # [B, L, 1, D]
            gate_scores.append(self.field_gates[i](emb).squeeze(-1).unsqueeze(-1))  # [B, L, 1]

        proj = torch.cat(proj_parts, dim=2)       # [B, L, F, D]
        scores = torch.cat(gate_scores, dim=2)    # [B, L, F]

        if valid_field_mask is not None:
            mask = valid_field_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~mask.view(1, 1, -1), -1e4)

        weights = torch.softmax(scores, dim=2)    # [B, L, F]
        fafe = (proj * weights.unsqueeze(-1)).sum(dim=2)  # [B, L, D]
        fafe = self.out_dropout(F.gelu(self.out_norm(fafe)))
        return base_token + torch.sigmoid(self.res_alpha) * fafe



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
        """Append sample-level calendar ids to the user-int feature stream."""
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
        use_dense_presence_flags: bool = True,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        use_alibi: bool = False,
        alibi_bias_scale: float = 1.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # Dormant optional experiments. Defaults preserve the original v24
        # behavior and checkpoint shape unless explicitly enabled.
        use_qgen_time_decay: bool = False,
        qgen_time_decay_strength: float = 0.0,
        use_target_aware_ns_gate: bool = False,
        # Per-domain encoder override: domains listed here use LongerEncoder
        # regardless of ``seq_encoder_type``. Unmatched names raise.
        longer_domains: Optional[List[str]] = None,
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
        self.use_dense_presence_flags = use_dense_presence_flags
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.use_alibi = use_alibi
        self.alibi_bias_scale = float(alibi_bias_scale)
        if use_rope and use_alibi:
            raise ValueError(
                "use_rope and use_alibi are mutually exclusive; choose one position encoding.")
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.use_qgen_time_decay = bool(use_qgen_time_decay)
        self.qgen_time_decay_strength = float(qgen_time_decay_strength)
        self.use_target_aware_ns_gate = bool(use_target_aware_ns_gate)
        self.sample_calendar_int_dim = (
            len(USER_CALENDAR_INT_VOCABS) if use_calendar_time_features else 0)
        self.sample_calendar_dense_dim = (
            USER_CALENDAR_DENSE_DIM if use_calendar_time_features else 0)

        user_int_feature_specs, user_ns_groups = self._append_calendar_int_specs(
            user_int_feature_specs, user_ns_groups, use_calendar_time_features)
        item_int_feature_specs, item_ns_groups = self._append_calendar_int_specs(
            item_int_feature_specs, item_ns_groups, use_calendar_time_features)

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
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            dense_presence_dim = 1 if use_dense_presence_flags else 0
            sum_dense_dim = USER_SUM_DENSE_RANGE[1] - USER_SUM_DENSE_RANGE[0]
            ads_dense_dim = USER_ADS_DENSE_RANGE[1] - USER_ADS_DENSE_RANGE[0]
            side_dense_dim = (
                USER_SIDE_DENSE_RANGE_A[1] - USER_SIDE_DENSE_RANGE_A[0]
                + USER_SIDE_DENSE_RANGE_B[1] - USER_SIDE_DENSE_RANGE_B[0]
                + self.sample_calendar_dense_dim
            )
            side_input_dim = (
                side_dense_dim * 2 if use_dense_presence_flags else side_dense_dim)
            self.user_dense_side_proj = nn.Sequential(
                nn.Linear(side_input_dim, d_model),
                nn.LayerNorm(d_model),
            )
            self.user_dense_sum_proj = nn.Sequential(
                nn.Linear(sum_dense_dim + dense_presence_dim, d_model),
                nn.LayerNorm(d_model),
            )
            self.user_dense_ads_proj = nn.Sequential(
                nn.Linear(ads_dense_dim + dense_presence_dim, d_model),
                nn.LayerNorm(d_model),
            )
            self.user_dense_merge_norm = nn.LayerNorm(d_model)
            expected_user_dense_dim = USER_SIDE_DENSE_RANGE_B[1]
            if user_dense_dim != expected_user_dense_dim:
                logging.warning(
                    f"user_dense_dim={user_dense_dim}, split dense projector "
                    f"uses the fixed {expected_user_dense_dim}-dim layout.")
        elif use_calendar_time_features:
            logging.warning(
                "use_calendar_time_features=True but user_dense_dim=0; "
                "sample-level time features are skipped to keep RankMixer T unchanged.")

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            item_dense_input_dim = (
                item_dense_dim * 2 if use_dense_presence_flags else item_dense_dim)
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_input_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Current item target token used by DIN query decoding. It is built
        # from item NS tokens, plus item dense token when that branch exists.
        self.item_target_num_tokens = num_item_ns + (1 if self.has_item_dense else 0)
        self.item_target_proj = nn.Sequential(
            nn.Linear(self.item_target_num_tokens * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        if self.use_target_aware_ns_gate:
            self.target_aware_ns_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            logging.info("Dormant experiment enabled: target-aware NS gate")

        # Total NS token count
        self.num_ns = (num_user_ns + (1 if self.has_user_dense else 0)
                       + num_item_ns + (1 if self.has_item_dense else 0))
        logging.info(
            "DIN item target: item_tokens=%s, d_model=%s",
            self.item_target_num_tokens, d_model,
        )

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

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
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
        # Preserve the high-score seq_d Field-Gate FAFE tokenization branch.
        # This only changes per-event seq_d token construction and does not
        # change NS token count or RankMixer full-mode T.
        self.fafe_domains = {'seq_d'}
        self._seq_fafe = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            if domain in self.fafe_domains:
                self._seq_fafe[domain] = FieldGateFAFE(
                    num_fields=len(vs),
                    emb_dim=emb_dim,
                    d_model=d_model,
                    dropout=dropout_rate,
                )
                logging.info(
                    "FieldGateFAFE enabled for %s: num_fields=%s, emb_dim=%s, d_model=%s",
                    domain, len(vs), emb_dim, d_model,
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
            use_time_decay=self.use_qgen_time_decay,
            time_decay_strength=self.qgen_time_decay_strength,
        )
        if self.use_qgen_time_decay:
            logging.info(
                "Dormant experiment enabled: QueryGenerator time-decay pooling, strength=%s",
                self.qgen_time_decay_strength,
            )

        # Resolve per-domain sequence encoder types. ``longer_domains`` wins
        # over the global ``seq_encoder_type``; everything else stays on the
        # global default. List ordering follows ``self.seq_domains``.
        longer_set = set(longer_domains or [])
        unknown_longer = longer_set - set(self.seq_domains)
        if unknown_longer:
            raise ValueError(
                f"longer_domains references unknown domains: {sorted(unknown_longer)}. "
                f"Available seq_domains: {self.seq_domains}")
        per_domain_encoder_types: List[str] = [
            'longer' if d in longer_set else seq_encoder_type
            for d in self.seq_domains
        ]
        if longer_set:
            logging.info(
                "Per-domain seq encoders: %s (LongerEncoder applied to %s, top_k=%s, causal=%s)",
                list(zip(self.seq_domains, per_domain_encoder_types)),
                sorted(longer_set), seq_top_k, seq_causal,
            )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=per_domain_encoder_types,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                num_time_buckets=num_time_buckets,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== Sequence Position Bias ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
            self.alibi_position_bias = None
            logging.info(
                "RoPE enabled: head_dim=%s, num_heads=%s, rope_base=%s. "
                "Position 0 is newest because sequence layout is newest-first.",
                head_dim, num_heads, rope_base,
            )
        else:
            self.rotary_emb = None
            logging.info("RoPE disabled")

        if use_alibi:
            self.alibi_position_bias = ALiBiPositionBias(
                num_heads=num_heads,
                bias_scale=alibi_bias_scale,
            )
            logging.info(
                "ALiBi enabled: num_heads=%s, bias_scale=%s, slopes=%s, "
                "distance=abs(q_pos-k_pos), position 0 is newest.",
                num_heads, alibi_bias_scale,
                [round(float(x), 8)
                 for x in self.alibi_position_bias.slopes.detach().cpu()],
            )
        else:
            self.alibi_position_bias = None
            logging.info("ALiBi disabled")

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

        # time_embedding and DIN time_score_bias are always preserved
        if self.num_time_buckets > 0:
            skip_count += 1
            for block in self.blocks:
                for din in block.din_attns:
                    if getattr(din, 'time_score_bias', None) is not None:
                        skip_count += 1
        if self.use_calendar_time_features:
            skip_count += 3

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

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
        """Append sample-level calendar ids to the user-int feature tensor."""
        if self.sample_calendar_int_dim == 0:
            return int_feats
        return torch.cat([int_feats, time_feats.long()], dim=-1)

    def _calendar_dense_features(
        self,
        time_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build bounded continuous calendar features for the user-dense path."""
        B = time_feats.shape[0]
        if self.sample_calendar_dense_dim == 0:
            empty = time_feats.new_zeros(B, 0, dtype=torch.float32)
            return empty, empty

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
        dense_time = dense_time * valid
        return dense_time, valid.expand(-1, USER_CALENDAR_DENSE_DIM)

    def _with_dense_presence(
        self,
        dense_feats: torch.Tensor,
        presence_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Append per-dimension dense presence flags when enabled."""
        if not self.use_dense_presence_flags:
            return dense_feats
        return torch.cat([dense_feats, presence_feats.to(dtype=dense_feats.dtype)], dim=-1)

    def _with_segment_presence(
        self,
        dense_feats: torch.Tensor,
        presence_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Append a single presence flag for a dense embedding block."""
        if not self.use_dense_presence_flags:
            return dense_feats
        segment_presence = presence_feats.amax(dim=-1, keepdim=True)
        return torch.cat(
            [dense_feats, segment_presence.to(dtype=dense_feats.dtype)], dim=-1)

    @staticmethod
    def _take_dense_range(
        dense_feats: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Take a fixed dense segment, zero-padding if the input is shorter."""
        dense_part = dense_feats[:, start:end]
        expected_dim = end - start
        if dense_part.shape[-1] == expected_dim:
            return dense_part
        return F.pad(dense_part, (0, expected_dim - dense_part.shape[-1]))

    def _build_user_dense_token(
        self,
        dense_feats: torch.Tensor,
        presence_feats: torch.Tensor,
        time_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Project SUM, LMF4Ads and other user dense blocks separately."""
        time_values, time_presence = self._calendar_dense_features(time_feats)
        sum_values = self._take_dense_range(dense_feats, *USER_SUM_DENSE_RANGE)
        side_values = torch.cat([
            self._take_dense_range(dense_feats, *USER_SIDE_DENSE_RANGE_A),
            self._take_dense_range(dense_feats, *USER_SIDE_DENSE_RANGE_B),
            time_values.to(dtype=dense_feats.dtype),
        ], dim=-1)
        ads_values = self._take_dense_range(dense_feats, *USER_ADS_DENSE_RANGE)

        sum_input = self._with_segment_presence(
            sum_values,
            self._take_dense_range(presence_feats, *USER_SUM_DENSE_RANGE))
        side_input = self._with_dense_presence(
            side_values,
            torch.cat([
                self._take_dense_range(presence_feats, *USER_SIDE_DENSE_RANGE_A),
                self._take_dense_range(presence_feats, *USER_SIDE_DENSE_RANGE_B),
                time_presence.to(dtype=presence_feats.dtype),
            ], dim=-1))
        ads_input = self._with_segment_presence(
            ads_values,
            self._take_dense_range(presence_feats, *USER_ADS_DENSE_RANGE))

        dense_token = (
            self.user_dense_side_proj(side_input)
            + self.user_dense_sum_proj(sum_input)
            + self.user_dense_ads_proj(ads_input)
        )
        dense_token = self.user_dense_merge_norm(dense_token)
        return F.silu(dense_token).unsqueeze(1)

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
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float32))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Optional Field-Gate FAFE: only enabled for selected domains, currently seq_d.
        # Keep time features after FAFE so temporal information remains a separate residual.
        if domain is not None and domain in self._seq_fafe:
            valid_field_mask = torch.tensor(
                [idx != -1 for idx in emb_index[:S]],
                dtype=torch.bool,
                device=token_emb.device,
            )
            token_emb = self._seq_fafe[domain](
                emb_list=emb_list,
                base_token=token_emb,
                valid_field_mask=valid_field_mask,
            )

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            time_emb = self.time_embedding(time_bucket_ids).to(dtype=token_emb.dtype)
            token_emb = token_emb + time_emb

        if self.use_calendar_time_features and calendar_time_feats is not None:
            calendar_time_emb = self._calendar_time_embedding(
                calendar_time_feats,
                self.seq_hour_embedding,
                self.seq_weekday_embedding,
                self.seq_period_embedding,
            ).to(dtype=token_emb.dtype)
            gate = torch.tanh(self.seq_calendar_time_gate).to(dtype=token_emb.dtype)
            token_emb = token_emb + gate * calendar_time_emb

        # Day-type embeddings: is_weekend + is_holiday
        if day_type_feats is not None:
            weekend_emb = self.seq_weekend_embedding(day_type_feats[..., 0].long())
            holiday_emb = self.seq_holiday_embedding(day_type_feats[..., 1].long())
            dt_emb = (weekend_emb + holiday_emb).to(dtype=token_emb.dtype)
            dt_gate = torch.tanh(self.seq_day_type_gate).to(dtype=token_emb.dtype)
            token_emb = token_emb + dt_gate * dt_emb

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _maybe_apply_target_aware_ns_gate(
        self,
        ns_tokens: torch.Tensor,
        target_item: torch.Tensor,
    ) -> torch.Tensor:
        """Optionally calibrate NS tokens with the current item target.

        This is a disabled-by-default experiment. When
        use_target_aware_ns_gate=False, this function is an exact identity and
        does not affect baseline training or inference.
        """
        if not self.use_target_aware_ns_gate:
            return ns_tokens
        target_expanded = target_item.unsqueeze(1).expand(-1, ns_tokens.shape[1], -1)
        gate_input = torch.cat([ns_tokens, target_expanded], dim=-1)
        gate = self.target_aware_ns_gate(gate_input)
        return ns_tokens * (0.5 + gate)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        target_item: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        seq_time_buckets_list: Optional[list] = None,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            target_item = self.emb_dropout(target_item)
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
                target_item=target_item,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                seq_time_buckets_list=seq_time_buckets_list,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
                alibi_position_bias=self.alibi_position_bias,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        user_int_feats = self._with_sample_calendar_int(
            inputs.user_int_feats, inputs.time_feats)
        user_ns = self.user_ns_tokenizer(user_int_feats)   # (B, num_user_groups, D)
        item_int_feats = self._with_sample_calendar_int(
            inputs.item_int_feats, inputs.time_feats)
        item_ns = self.item_ns_tokenizer(item_int_feats)

        ns_parts = [user_ns]
        item_target_parts = [item_ns]
        if self.has_user_dense:
            user_dense_tok = self._build_user_dense_token(
                inputs.user_dense_feats,
                inputs.user_dense_presence_feats,
                inputs.time_feats)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_in = self._with_dense_presence(
                inputs.item_dense_feats, inputs.item_dense_presence_feats)
            item_dense_tok = F.silu(self.item_dense_proj(item_dense_in)).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(item_dense_tok)
            item_target_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)
        target_item = self.item_target_proj(
            torch.cat(item_target_parts, dim=1).reshape(item_ns.shape[0], -1)
        )  # (B, D)
        ns_tokens = self._maybe_apply_target_aware_ns_gate(ns_tokens, target_item)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        seq_time_buckets_list = []
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
            seq_time_buckets_list.append(inputs.seq_time_buckets[domain])
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list,
            seq_time_buckets_list=seq_time_buckets_list,
        )

        # 4. Dropout + MultiSeqHyFormerBlock stack + output projection
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, target_item, seq_tokens_list, seq_masks_list,
            seq_time_buckets_list=seq_time_buckets_list,
            apply_dropout=self.training
        )

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        user_int_feats = self._with_sample_calendar_int(
            inputs.user_int_feats, inputs.time_feats)
        user_ns = self.user_ns_tokenizer(user_int_feats)
        item_int_feats = self._with_sample_calendar_int(
            inputs.item_int_feats, inputs.time_feats)
        item_ns = self.item_ns_tokenizer(item_int_feats)

        ns_parts = [user_ns]
        item_target_parts = [item_ns]
        if self.has_user_dense:
            user_dense_tok = self._build_user_dense_token(
                inputs.user_dense_feats,
                inputs.user_dense_presence_feats,
                inputs.time_feats)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_in = self._with_dense_presence(
                inputs.item_dense_feats, inputs.item_dense_presence_feats)
            item_dense_tok = F.silu(self.item_dense_proj(item_dense_in)).unsqueeze(1)
            ns_parts.append(item_dense_tok)
            item_target_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)
        target_item = self.item_target_proj(
            torch.cat(item_target_parts, dim=1).reshape(item_ns.shape[0], -1)
        )
        ns_tokens = self._maybe_apply_target_aware_ns_gate(ns_tokens, target_item)

        seq_tokens_list = []
        seq_masks_list = []
        seq_time_buckets_list = []
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
            seq_time_buckets_list.append(inputs.seq_time_buckets[domain])
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list,
            seq_time_buckets_list=seq_time_buckets_list,
        )

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, target_item, seq_tokens_list, seq_masks_list,
            seq_time_buckets_list=seq_time_buckets_list,
            apply_dropout=False
        )

        logits = self.clsfier(output)
        return logits, output
