import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq


ITEM_DENSE_FEATURE_NAMES: List[str] = [
    "item_hist_cnt_all_log1p",
    "item_hist_active_hour_nuniq_log1p",
    "item_hist_active_day_nuniq",
    "item_hist_peak_hour_ratio",
    "item_hist_alive_hours_log1p",
    "item_hist_recency_hours_log1p",
    "item_hist_cnt_1h_log1p",
    "item_hist_cnt_6h_log1p",
    "item_hist_cnt_1d_log1p",
    "item_hist_trend_cnt_1h_1d",
    "item_hist_hour_entropy",
    "item_hist_day_entropy",
]

# Synthetic fids reserved for online-built item dense features.
ITEM_DENSE_FEATURE_FIDS: List[int] = [
    10000 + i for i in range(1, len(ITEM_DENSE_FEATURE_NAMES) + 1)
]

_HOUR_SECONDS = 3600
_DAY_SECONDS = 86400
_WINDOW_SPECS: Mapping[str, int] = {
    "1h": 1,
    "6h": 6,
    "1d": 24,
}


@dataclass
class ItemFeatureTable:
    feature_names: Sequence[str]
    feature_fids: Sequence[int]
    vectors_by_item: Dict[int, np.ndarray]


@dataclass
class _ItemCoreStats:
    total_count: int = 0
    first_ts: Optional[int] = None
    last_ts: Optional[int] = None


def _safe_log1p(x: float) -> float:
    return float(math.log1p(max(x, 0.0)))


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _entropy_from_counter(counter: Mapping[int, int]) -> float:
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            ent -= p * math.log(p)
    return float(ent)


def _group_row_groups(
    row_groups: Sequence[Tuple[str, int, int]]
) -> Mapping[str, List[int]]:
    grouped: MutableMapping[str, List[int]] = defaultdict(list)
    for file_path, row_group_idx, _ in row_groups:
        grouped[file_path].append(int(row_group_idx))
    return grouped


class ItemDenseFeatureBuilder:
    """Builds scalable item-side dense features from interaction logs only.

    Features are aggregated once per dataset split, producing a lookup table
    ``item_id -> dense vector``. This design keeps the runtime batch path cheap
    and avoids cross-worker shared-state issues in multi-process DataLoaders.
    """

    def __init__(
        self,
        row_groups: Sequence[Tuple[str, int, int]],
        batch_size: int = 65536,
        timestamp_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> None:
        self._row_groups = list(row_groups)
        self._batch_size = batch_size
        self._timestamp_range = timestamp_range

    def _filter_by_timestamp(
        self,
        item_ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._timestamp_range is None:
            return item_ids, timestamps
        lo, hi = self._timestamp_range
        mask = np.ones(len(timestamps), dtype=bool)
        if lo is not None:
            mask &= timestamps > int(lo)
        if hi is not None:
            mask &= timestamps <= int(hi)
        return item_ids[mask], timestamps[mask]

    def build(self) -> ItemFeatureTable:
        core_stats: MutableMapping[int, _ItemCoreStats] = {}
        hour_total_counts: MutableMapping[int, Counter] = defaultdict(Counter)
        day_total_counts: MutableMapping[int, Counter] = defaultdict(Counter)
        max_ts: Optional[int] = None

        grouped_row_groups = _group_row_groups(self._row_groups)
        for file_path, rg_indices in grouped_row_groups.items():
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(
                columns=["item_id", "timestamp"],
                batch_size=self._batch_size,
                row_groups=rg_indices,
            ):
                item_ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
                timestamps = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
                item_ids, timestamps = self._filter_by_timestamp(item_ids, timestamps)

                if len(item_ids) == 0:
                    continue

                batch_max_ts = int(timestamps.max())
                max_ts = batch_max_ts if max_ts is None else max(max_ts, batch_max_ts)

                sort_idx = np.argsort(item_ids, kind="stable")
                s_item_ids = item_ids[sort_idx]
                s_timestamps = timestamps[sort_idx]

                uniq_items, starts = np.unique(s_item_ids, return_index=True)
                ends = np.append(starts[1:], len(s_item_ids))
                counts = ends - starts
                min_ts = np.minimum.reduceat(s_timestamps, starts)
                max_item_ts = np.maximum.reduceat(s_timestamps, starts)

                for item_id, count, item_min_ts, item_max_ts in zip(
                    uniq_items,
                    counts,
                    min_ts,
                    max_item_ts,
                ):
                    item_id_i = int(item_id)
                    stat = core_stats.get(item_id_i)
                    if stat is None:
                        stat = _ItemCoreStats()
                        core_stats[item_id_i] = stat
                    stat.total_count += int(count)
                    cur_min_ts = int(item_min_ts)
                    cur_max_ts = int(item_max_ts)
                    stat.first_ts = (
                        cur_min_ts
                        if stat.first_ts is None
                        else min(stat.first_ts, cur_min_ts)
                    )
                    stat.last_ts = (
                        cur_max_ts
                        if stat.last_ts is None
                        else max(stat.last_ts, cur_max_ts)
                    )

                hour_buckets = timestamps // _HOUR_SECONDS
                day_buckets = timestamps // _DAY_SECONDS

                hour_pairs = np.column_stack((item_ids, hour_buckets))
                uniq_hour_pairs, hour_pair_counts = np.unique(
                    hour_pairs, axis=0, return_counts=True
                )
                for pair, count in zip(uniq_hour_pairs, hour_pair_counts):
                    item_id_i = int(pair[0])
                    hour_bucket_i = int(pair[1])
                    hour_total_counts[item_id_i][hour_bucket_i] += int(count)

                day_pairs = np.column_stack((item_ids, day_buckets))
                uniq_day_pairs, day_pair_counts = np.unique(
                    day_pairs, axis=0, return_counts=True
                )
                for pair, count in zip(uniq_day_pairs, day_pair_counts):
                    item_id_i = int(pair[0])
                    day_bucket_i = int(pair[1])
                    day_total_counts[item_id_i][day_bucket_i] += int(count)

        if max_ts is None:
            return ItemFeatureTable(
                feature_names=ITEM_DENSE_FEATURE_NAMES,
                feature_fids=ITEM_DENSE_FEATURE_FIDS,
                vectors_by_item={},
            )

        max_hour_bucket = max_ts // _HOUR_SECONDS
        vectors_by_item: Dict[int, np.ndarray] = {}
        for item_id, stat in core_stats.items():
            total_count = float(stat.total_count)
            first_ts = int(stat.first_ts or max_ts)
            last_ts = int(stat.last_ts or max_ts)

            item_hour_counts = hour_total_counts.get(item_id, Counter())
            item_day_counts = day_total_counts.get(item_id, Counter())

            cnt_windows: Dict[str, float] = {}
            for window_name, window_hours in _WINDOW_SPECS.items():
                min_hour = max_hour_bucket - (window_hours - 1)
                cnt_windows[window_name] = float(
                    sum(
                        count
                        for hour_bucket, count in item_hour_counts.items()
                        if hour_bucket >= min_hour
                    )
                )

            peak_hour_cnt = float(max(item_hour_counts.values())) if item_hour_counts else 0.0
            alive_hours = max(0.0, (last_ts - first_ts) / float(_HOUR_SECONDS))
            recency_hours = max(0.0, (max_ts - last_ts) / float(_HOUR_SECONDS))

            vector = np.array(
                [
                    _safe_log1p(total_count),
                    _safe_log1p(float(len(item_hour_counts))),
                    float(len(item_day_counts)),
                    _safe_ratio(peak_hour_cnt, total_count),
                    _safe_log1p(alive_hours),
                    _safe_log1p(recency_hours),
                    _safe_log1p(cnt_windows["1h"]),
                    _safe_log1p(cnt_windows["6h"]),
                    _safe_log1p(cnt_windows["1d"]),
                    _safe_ratio(cnt_windows["1h"], cnt_windows["1d"]),
                    _entropy_from_counter(item_hour_counts),
                    _entropy_from_counter(item_day_counts),
                ],
                dtype=np.float32,
            )
            vectors_by_item[item_id] = vector

        return ItemFeatureTable(
            feature_names=ITEM_DENSE_FEATURE_NAMES,
            feature_fids=ITEM_DENSE_FEATURE_FIDS,
            vectors_by_item=vectors_by_item,
        )
