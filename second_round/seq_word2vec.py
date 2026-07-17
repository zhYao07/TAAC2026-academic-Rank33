"""In-job word2vec (item2vec) warm-start for high-cardinality sequence ids (v28).

This runs INSIDE the single training job (train.py): it trains a gensim
word2vec on the behavior-sequence columns, then hands back, per id feature, a
``(kept_ids, init_matrix)`` pair used to initialize a compressed-vocab
embedding inside the model. The learned (fine-tuned) embeddings are saved in
the one ``model.pt``; no word2vec artifact is needed at inference.

gensim is imported lazily inside ``build_seq_w2v`` so importing this module
(e.g. from the model at inference time) never requires gensim.
"""

import os
import tempfile
import logging
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _write_sentences(
    parquet_files: List[str],
    column: str,
    out_fh,
    max_sentences: int,
    max_seq_len: int,
    read_batch: int = 4096,
) -> int:
    """Stream the sequence column to a gensim corpus file (one sentence/line).

    Each row's id list (values > 0, truncated to ``max_seq_len``) becomes one
    whitespace-separated line of integer tokens. Returns the number of
    sentences written (capped at ``max_sentences``).
    """
    written = 0
    for fpath in parquet_files:
        pf = pq.ParquetFile(fpath)
        if column not in pf.schema_arrow.names:
            continue
        for batch in pf.iter_batches(batch_size=read_batch, columns=[column]):
            col = batch.column(0)
            if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                offs = col.offsets.to_numpy()
                vals = col.values.to_numpy(zero_copy_only=False)
                rows = (vals[int(offs[i]):int(offs[i + 1])]
                        for i in range(len(offs) - 1))
            else:
                vals = col.fill_null(0).to_numpy(zero_copy_only=False)
                rows = (np.asarray([v]) for v in vals)

            for row in rows:
                if row.size == 0:
                    continue
                row = row[row > 0]
                if row.size == 0:
                    continue
                if row.size > max_seq_len:
                    row = row[:max_seq_len]
                out_fh.write(" ".join(map(str, row.tolist())))
                out_fh.write("\n")
                written += 1
                if written >= max_sentences:
                    return written
    return written


def build_seq_w2v(
    parquet_files: List[str],
    column: str,
    emb_dim: int = 64,
    top_k: int = 2_000_000,
    min_count: int = 5,
    window: int = 5,
    epochs: int = 3,
    workers: int = 8,
    max_sentences: int = 2_000_000,
    max_seq_len: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train word2vec on one sequence column; return ``(kept_ids, init_matrix)``.

    ``kept_ids``: ascending int64 array of the top-``top_k`` most frequent ids
    (after ``min_count`` filtering), suitable for ``searchsorted``.
    ``init_matrix``: float32 ``[len(kept_ids)+1, emb_dim]``; row 0 is the OOV /
    padding zero vector, rows ``1..K`` are the word2vec vectors of ``kept_ids``.
    Returns empty arrays if no token survives.
    """
    from gensim.models import Word2Vec  # lazy: only the in-job step needs gensim

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".w2v.txt", delete=False, encoding="utf-8")
    tmp_path = tmp.name
    try:
        n = _write_sentences(
            parquet_files, column, tmp, max_sentences, max_seq_len)
        tmp.close()
        if n == 0:
            logging.warning("seq_w2v[%s]: no sentences, skipping", column)
            return (np.zeros(0, dtype=np.int64),
                    np.zeros((1, emb_dim), dtype=np.float32))

        model = Word2Vec(
            corpus_file=tmp_path,
            vector_size=emb_dim,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,
            epochs=epochs,
            seed=seed,
        )
        wv = model.wv
        # index_to_key is ordered by descending frequency in gensim 4.x.
        tokens = wv.index_to_key[:top_k]
        kept = np.array(sorted(int(t) for t in tokens), dtype=np.int64)
        mat = np.zeros((kept.shape[0] + 1, emb_dim), dtype=np.float32)
        for j, tid in enumerate(kept):
            mat[j + 1] = wv[str(int(tid))]
        logging.info(
            "seq_w2v[%s]: sentences=%d, vocab=%d, kept=%d (top_k=%d, min_count=%d)",
            column, n, len(wv.index_to_key), kept.shape[0], top_k, min_count)
        return kept, mat
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def build_all_seq_w2v(
    parquet_files: List[str],
    specs: List[Tuple[str, str, int]],
    **kw,
) -> Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]:
    """Train word2vec for several id features.

    Args:
        parquet_files: list of parquet paths.
        specs: list of ``(domain, prefix, fid)``; the column read is
            ``f"{prefix}_{fid}"``.
        **kw: forwarded to :func:`build_seq_w2v`.

    Returns:
        ``{(domain, fid): (kept_ids, init_matrix)}``.
    """
    out = {}
    for domain, prefix, fid in specs:
        column = "%s_%d" % (prefix, fid)
        kept, mat = build_seq_w2v(parquet_files, column, **kw)
        out[(domain, int(fid))] = (kept, mat)
    return out


def build_all_column_w2v(
    parquet_files: List[str],
    specs: List[Tuple[str, int, str]],
    **kw,
) -> Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]:
    """Train word2vec for arbitrary int/list columns.

    Args:
        parquet_files: list of parquet paths.
        specs: list of ``(side, fid, column_name)``.
        **kw: forwarded to :func:`build_seq_w2v`.

    Returns:
        ``{(side, fid): (kept_ids, init_matrix)}``.
    """
    out = {}
    for side, fid, column in specs:
        kept, mat = build_seq_w2v(parquet_files, column, **kw)
        out[(side, int(fid))] = (kept, mat)
    return out
