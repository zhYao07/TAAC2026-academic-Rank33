"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1
DENSE_STATS_FILENAME = 'dense_stats.json'
DENSE_STD_EPS = 1e-6
USER_DENSE_SUM_RANGE = (0, 256)
USER_DENSE_ADS_RANGE = (568, 888)

# Calendar-time categorical features derived from Unix-second timestamps.
# Id 0 is reserved for padding / missing values.
TIME_FEAT_DIM = 3
DAY_TYPE_DIM = 2  # [is_weekend, is_holiday]

# Chinese public holidays 2025-2026 (official/estimated).
_CN_HOLIDAYS = frozenset([
    # ---- 2025 ----
    (2025, 1, 1),
    *((2025, 1, d) for d in range(28, 32)), *((2025, 2, d) for d in range(1, 5)),  # Spring Festival
    (2025, 4, 4), (2025, 4, 5), (2025, 4, 6),             # Qingming
    *((2025, 5, d) for d in range(1, 6)),                   # Labor Day
    (2025, 5, 31), (2025, 6, 1), (2025, 6, 2),             # Dragon Boat
    *((2025, 10, d) for d in range(1, 9)),                  # National Day
    # ---- 2026 ----
    (2026, 1, 1), (2026, 1, 2), (2026, 1, 3),
    *((2026, 2, d) for d in range(15, 22)),                 # Spring Festival (CNY = Feb 17)
    (2026, 4, 4), (2026, 4, 5), (2026, 4, 6),             # Qingming
    *((2026, 5, d) for d in range(1, 6)),                   # Labor Day
    (2026, 6, 19), (2026, 6, 20), (2026, 6, 21),          # Dragon Boat
    (2026, 9, 25), (2026, 9, 26), (2026, 9, 27),          # Mid-Autumn
    *((2026, 10, d) for d in range(1, 9)),                  # National Day
])
NUM_HOUR_IDS = 25       # 0=padding, 1..24 represent hour 0..23
NUM_WEEKDAY_IDS = 8     # 0=padding, 1..7 represent Monday..Sunday
NUM_PERIOD_IDS = 7      # 0=padding, 1..6 coarse daily periods
DEFAULT_TIME_TZ_OFFSET_HOURS = 8.0
TIME_PERIOD_HOUR_BOUNDARIES = np.array([6, 11, 14, 18, 22], dtype=np.int64)


def fill_calendar_time_features(
    timestamps: "npt.NDArray[np.int64]",
    out: "npt.NDArray[np.int64]",
    tz_offset_seconds: int,
) -> "npt.NDArray[np.int64]":
    """Fill ``out`` with [hour_id, weekday_id, period_id] from timestamps.

    ``timestamps`` is assumed to be Unix seconds, matching the existing
    time-delta bucket logic. A timestamp <= 0 is treated as padding/missing and
    leaves all three feature ids at 0.
    """
    out[...] = 0
    valid = timestamps > 0
    if not valid.any():
        return out

    local_ts = timestamps[valid].astype(np.int64, copy=False) + tz_offset_seconds
    hours = (local_ts // 3600) % 24
    days = local_ts // 86400

    hour_ids = out[..., 0]
    weekday_ids = out[..., 1]
    period_ids = out[..., 2]

    hour_ids[valid] = hours + 1
    # 1970-01-01 was Thursday. Monday=1, ..., Sunday=7.
    weekday_ids[valid] = ((days + 3) % 7) + 1
    period_ids[valid] = (
        np.searchsorted(TIME_PERIOD_HOUR_BOUNDARIES, hours, side='right') + 1
    )
    return out


def fill_day_type_features(
    timestamps: "npt.NDArray[np.int64]",
    out: "npt.NDArray[np.int64]",
    tz_offset_seconds: int,
) -> "npt.NDArray[np.int64]":
    """Fill ``out`` with [is_weekend, is_holiday] from timestamps.

    Values: 0 = padding, 1 = no, 2 = yes.
    """
    out[...] = 0
    valid = timestamps > 0
    if not valid.any():
        return out

    local_ts = timestamps[valid].astype(np.int64, copy=False) + tz_offset_seconds
    days = local_ts // 86400

    # is_weekend: epoch 1970-01-01 was Thursday (weekday=3).
    # 0=Mon..6=Sun; 5,6 = Sat,Sun.
    weekday = (days + 3) % 7  # 0=Mon, 6=Sun
    is_weekend = out[..., 0]
    is_weekend[valid] = np.where(weekday >= 5, 2, 1)  # 2=yes, 1=no

    # is_holiday: lookup in _CN_HOLIDAYS set.
    import pandas as pd
    dt_idx = pd.to_datetime(local_ts, unit='s', utc=True)
    years = dt_idx.year.to_numpy()
    months = dt_idx.month.to_numpy()
    day_of_month = dt_idx.day.to_numpy()
    holiday_flags = np.array(
        [(int(y), int(m), int(d)) in _CN_HOLIDAYS for y, m, d in zip(years, months, day_of_month)],
        dtype=np.int64,
    )
    is_holiday = out[..., 1]
    is_holiday[valid] = np.where(holiday_flags, 2, 1)  # 2=yes, 1=no

    return out


def signed_log1p_transform(x: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
    """Signed log1p transform that supports negative dense values."""
    return np.sign(x) * np.log1p(np.abs(x))


def build_user_dense_value_transform_mask(total_dim: int) -> "npt.NDArray[np.bool_]":
    """Mask dense dimensions that should receive value normalization.

    Pre-aggregated embedding blocks keep their original geometry; the log1p
    and z-score path is reserved for ordinary scalar/statistical dense values.
    """
    mask = np.ones(total_dim, dtype=np.bool_)
    for start, end in [USER_DENSE_SUM_RANGE, USER_DENSE_ADS_RANGE]:
        if start < total_dim:
            mask[start:min(end, total_dim)] = False
    return mask


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        time_tz_offset_hours: float = DEFAULT_TIME_TZ_OFFSET_HOURS,
        use_dense_value_norm: bool = True,
        use_dense_value_log1p: bool = True,
        dense_stats_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.time_tz_offset_seconds = int(round(time_tz_offset_hours * 3600))
        self.use_dense_value_norm = use_dense_value_norm
        self.use_dense_value_log1p = use_dense_value_log1p
        self._dense_mean: Optional["npt.NDArray[np.float32]"] = None
        self._dense_std: Optional["npt.NDArray[np.float32]"] = None
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})
        self._dense_value_transform_mask = build_user_dense_value_transform_mask(
            self.user_dense_schema.total_dim)

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_user_dense_presence = np.zeros(
            (B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_time_feats = np.zeros((B, TIME_FEAT_DIM), dtype=np.int64)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_tf = {}
        self._buf_seq_dt = {}  # day-type features (is_weekend, is_holiday)
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_tf[domain] = np.zeros((B, max_len, TIME_FEAT_DIM), dtype=np.int64)
            self._buf_seq_dt[domain] = np.zeros((B, max_len, DAY_TYPE_DIM), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

        if dense_stats_path is not None:
            self.load_dense_value_stats(dense_stats_path)

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        padded, _ = self._pad_varlen_float_column_with_presence(
            arrow_col, max_dim, B)
        return padded

    def _pad_varlen_float_column_with_presence(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.float32]", "npt.NDArray[np.float32]"]:
        """Pad a float list column and mark which dense positions exist."""
        offsets = arrow_col.offsets.to_numpy()
        values_array = arrow_col.values
        values = values_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.float32)
        value_valid = values_array.is_valid().to_numpy(zero_copy_only=False)
        list_valid = arrow_col.is_valid().to_numpy(zero_copy_only=False)

        padded = np.zeros((B, max_dim), dtype=np.float32)
        presence = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            if not list_valid[i]:
                continue
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            vals = values[start:start + use_len]
            finite = np.isfinite(vals)
            padded[i, :use_len] = np.where(finite, vals, 0.0)
            presence[i, :use_len] = (
                value_valid[start:start + use_len] & finite
            ).astype(np.float32)

        return padded, presence

    def _read_user_dense_batch(
        self, batch: "pa.RecordBatch"
    ) -> Tuple["npt.NDArray[np.float32]", "npt.NDArray[np.float32]"]:
        """Read raw user dense values and per-position presence flags."""
        B = batch.num_rows
        user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        user_dense_presence = np.zeros_like(user_dense)
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded, presence = self._pad_varlen_float_column_with_presence(col, dim, B)
            user_dense[:, offset:offset + dim] = padded
            user_dense_presence[:, offset:offset + dim] = presence
        return user_dense, user_dense_presence

    def set_dense_value_stats(
        self,
        mean: "npt.NDArray[np.float32]",
        std: "npt.NDArray[np.float32]",
    ) -> None:
        """Attach dense-feature normalization statistics to this dataset."""
        expected_dim = self.user_dense_schema.total_dim
        if len(mean) != expected_dim or len(std) != expected_dim:
            raise ValueError(
                f"Dense stats dim mismatch: expected {expected_dim}, "
                f"got mean={len(mean)}, std={len(std)}")
        self._dense_mean = np.asarray(mean, dtype=np.float32)
        self._dense_std = np.maximum(
            np.asarray(std, dtype=np.float32), DENSE_STD_EPS)

    def load_dense_value_stats(self, path: str) -> None:
        """Load dense-feature transform statistics from ``dense_stats.json``."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"dense stats file not found at {path}")
        with open(path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        self.set_dense_value_stats(
            np.asarray(stats['mean'], dtype=np.float32),
            np.asarray(stats['std'], dtype=np.float32),
        )
        logging.info(f"Loaded dense value stats from {path}")

    def save_dense_value_stats(self, path: str, stats: Dict[str, Any]) -> None:
        """Persist dense-feature transform statistics as JSON."""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Saved dense value stats to {path}")

    def compute_dense_value_stats(
        self,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute per-dimension stats for user dense values on this split.

        Statistics are computed over observed dense positions only. Padded or
        null positions keep value 0 in batches but do not bias mean/std.
        """
        dim = self.user_dense_schema.total_dim
        transform_mask = self._dense_value_transform_mask
        transform_mask_2d = transform_mask.reshape(1, -1)
        sum_x = np.zeros(dim, dtype=np.float64)
        sum_x2 = np.zeros(dim, dtype=np.float64)
        count = np.zeros(dim, dtype=np.float64)

        for file_path, rg_idx, _ in self._rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                dense, present = self._read_user_dense_batch(batch)
                if self.use_dense_value_log1p:
                    dense[:, transform_mask] = signed_log1p_transform(
                        dense[:, transform_mask])
                observed = (present > 0) & transform_mask_2d
                dense64 = dense.astype(np.float64, copy=False)
                sum_x += np.where(observed, dense64, 0.0).sum(axis=0)
                sum_x2 += np.where(observed, dense64 * dense64, 0.0).sum(axis=0)
                count += observed.sum(axis=0)

        mean = np.zeros(dim, dtype=np.float64)
        np.divide(sum_x, count, out=mean, where=count > 0)
        second_moment = np.zeros(dim, dtype=np.float64)
        np.divide(sum_x2, count, out=second_moment, where=count > 0)
        var = np.maximum(second_moment - mean * mean, 0.0)
        std = np.sqrt(var)
        std[(count <= 1) | (std < DENSE_STD_EPS)] = 1.0
        mean[~transform_mask] = 0.0
        std[~transform_mask] = 1.0

        stats = {
            'feature': 'user_dense',
            'num_dims': int(dim),
            'use_dense_value_log1p': bool(self.use_dense_value_log1p),
            'stat_scope': 'observed_dense_positions_excluding_embedding_blocks',
            'excluded_ranges': {
                'sum_embedding': list(USER_DENSE_SUM_RANGE),
                'ads_embedding': list(USER_DENSE_ADS_RANGE),
            },
            'mean': mean.astype(float).tolist(),
            'std': std.astype(float).tolist(),
            'count': count.astype(np.int64).tolist(),
        }
        self.set_dense_value_stats(
            np.asarray(stats['mean'], dtype=np.float32),
            np.asarray(stats['std'], dtype=np.float32),
        )
        if output_path:
            self.save_dense_value_stats(output_path, stats)
        logging.info(
            f"Computed dense value stats: dims={dim}, "
            f"observed_min={int(count[transform_mask].min()) if transform_mask.any() else 0}, "
            f"observed_max={int(count[transform_mask].max()) if transform_mask.any() else 0}, "
            f"use_log1p={self.use_dense_value_log1p}")
        return stats

    def _transform_user_dense_inplace(
        self,
        user_dense: "npt.NDArray[np.float32]",
        user_dense_presence: "npt.NDArray[np.float32]",
    ) -> None:
        """Apply configured dense value transform to a raw dense batch.

        Missing / padded dense positions are reset to 0 after normalization so
        the value branch stays neutral and missingness is represented only by
        the accompanying presence flags.
        """
        if not self.use_dense_value_norm:
            return
        transform_mask = self._dense_value_transform_mask
        if self.use_dense_value_log1p:
            user_dense[:, transform_mask] = signed_log1p_transform(
                user_dense[:, transform_mask])
        if self._dense_mean is None or self._dense_std is None:
            raise RuntimeError(
                "Dense value normalization is enabled but stats were not loaded. "
                "Call compute_dense_value_stats() or provide dense_stats_path.")
        user_dense[:, transform_mask] -= self._dense_mean[transform_mask]
        user_dense[:, transform_mask] /= self._dense_std[transform_mask]
        user_dense[user_dense_presence <= 0] = 0.0

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        time_feats = self._buf_time_feats[:B]
        fill_calendar_time_features(
            timestamps, time_feats, self.time_tz_offset_seconds)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense, user_dense_presence = self._read_user_dense_batch(batch)
        self._buf_user_dense[:B] = user_dense
        self._buf_user_dense_presence[:B] = user_dense_presence
        user_dense = self._buf_user_dense[:B]
        user_dense_presence = self._buf_user_dense_presence[:B]
        self._transform_user_dense_inplace(user_dense, user_dense_presence)

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'user_dense_presence_feats': torch.from_numpy(user_dense_presence.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'item_dense_presence_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'time_feats': torch.from_numpy(time_feats.copy()),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            seq_time_feats = self._buf_seq_tf[domain][:B]
            seq_time_feats[:] = 0
            seq_day_type = self._buf_seq_dt[domain][:B]
            seq_day_type[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets
                fill_calendar_time_features(
                    ts_padded, seq_time_feats, self.time_tz_offset_seconds)
                fill_day_type_features(
                    ts_padded, seq_day_type, self.time_tz_offset_seconds)

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            result[f'{domain}_time_feats'] = torch.from_numpy(seq_time_feats.copy())
            result[f'{domain}_day_type_feats'] = torch.from_numpy(seq_day_type.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    time_tz_offset_hours: float = DEFAULT_TIME_TZ_OFFSET_HOURS,
    use_dense_value_norm: bool = True,
    use_dense_value_log1p: bool = True,
    dense_stats_path: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        time_tz_offset_hours=time_tz_offset_hours,
        use_dense_value_norm=use_dense_value_norm,
        use_dense_value_log1p=use_dense_value_log1p,
    )

    if use_dense_value_norm:
        logging.info("Computing dense value normalization stats from training Row Groups")
        train_dataset.compute_dense_value_stats(dense_stats_path)

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        time_tz_offset_hours=time_tz_offset_hours,
        use_dense_value_norm=use_dense_value_norm,
        use_dense_value_log1p=use_dense_value_log1p,
        dense_stats_path=dense_stats_path,
    )
    if use_dense_value_norm and dense_stats_path is None:
        valid_dataset.set_dense_value_stats(
            train_dataset._dense_mean, train_dataset._dense_std)
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset
