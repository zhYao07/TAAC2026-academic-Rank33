"""PCVRHyFormer training entry point (self-contained baseline).

Supports single-GPU and multi-GPU DDP training (via torchrun).

Usage:
    # Single GPU
    python train.py [--num_epochs 10] [--batch_size 256] ...
    # Multi-GPU
    torchrun --nproc_per_node=N train.py [--num_epochs 10] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import time
import argparse
import logging
import hashlib
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


# ─────────────────────────── DDP Helpers ──────────────────────────────────


def setup_ddp():
    """Initialize DDP. Detects torchrun environment variables.

    Returns:
        (rank, local_rank, world_size). Non-DDP returns (0, 0, 1).
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # Long timeout: tolerates the in-job word2vec step and other rare long
        # gaps between collectives without tripping the NCCL watchdog.
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ─────────────────────────────────────────────────────────────────────────


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def build_seq_w2v_cache_path(
    args: argparse.Namespace,
    schema_path: str,
    parquet_files: List[str],
    specs: List[Tuple[str, str, int]],
) -> Optional[str]:
    """Return a deterministic USER_CACHE_PATH cache path for seq word2vec init."""
    cache_root = os.environ.get('USER_CACHE_PATH')
    if not cache_root:
        return None

    file_fingerprint = []
    for fpath in parquet_files:
        try:
            st = os.stat(fpath)
            file_fingerprint.append({
                'path': os.path.abspath(fpath),
                'size': int(st.st_size),
                'mtime_ns': int(st.st_mtime_ns),
            })
        except OSError:
            file_fingerprint.append({'path': os.path.abspath(fpath)})

    try:
        schema_stat = os.stat(schema_path)
        schema_fingerprint = {
            'path': os.path.abspath(schema_path),
            'size': int(schema_stat.st_size),
            'mtime_ns': int(schema_stat.st_mtime_ns),
        }
    except OSError:
        schema_fingerprint = {'path': os.path.abspath(schema_path)}

    payload = {
        'version': 1,
        'specs': specs,
        'schema': schema_fingerprint,
        'parquet_files': file_fingerprint,
        'emb_dim': int(args.emb_dim),
        'top_k': int(args.seq_w2v_top_k),
        'min_count': int(args.seq_w2v_min_count),
        'window': int(args.seq_w2v_window),
        'epochs': int(args.seq_w2v_epochs),
        'max_sentences': int(args.seq_w2v_max_sentences),
        'max_seq_len': int(args.seq_w2v_max_seq_len),
        'seed': int(args.seed),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')
    ).hexdigest()[:24]
    return os.path.join(cache_root, f'seq_w2v_{digest}.npz')


def save_npz_atomic(path: str, arrays: dict) -> None:
    """Atomically write an npz so DDP peers never observe a partial file."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + '.',
        suffix='.tmp.npz',
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--split_by_time', action='store_true',
                        help='Split train/valid by timestamp instead of Row Group position')
    parser.add_argument('--time_tz_offset_hours', type=float, default=8.0,
                        help='Timezone offset used to derive hour/weekday/timeperiod '
                             'from Unix-second timestamps (default: 8 for Asia/Shanghai).')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--save_every_epoch', action='store_true', default=False,
                        help='Save a self-contained checkpoint after every epoch '
                             '(auto-on when valid_ratio<=0 / full-data training).')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--use_calendar_time_features', action='store_true', default=True,
                        help='Enable sample-level and sequence-level hour/weekday/timeperiod '
                             'features derived from timestamps (default on).')
    parser.add_argument('--no_calendar_time_features',
                        dest='use_calendar_time_features', action='store_false',
                        help='Disable timestamp-derived calendar time features')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='Decay for EMA over DENSE params only (sparse embeddings '
                             'excluded). Eval/checkpoint use the EMA weights. '
                             '0 disables EMA. Typical: 0.9995.')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')
    parser.add_argument('--seq_hash_bucket_size', type=int, default=0,
                        help='Hash bucket size for sequence fields skipped by '
                             'emb_skip_threshold. 0 = disabled. Recommended: 500000.')
    parser.add_argument('--seq_hash_gate_init', type=float, default=-0.75,
                        help='Initial gate logit for sequence hash embeddings.')
    parser.add_argument('--ns_hash_bucket_size', type=int, default=0,
                        help='Hash bucket size for user/item int fields skipped by '
                             'emb_skip_threshold. 0 = disabled.')
    parser.add_argument('--ns_hash_gate_init', type=float, default=-0.75,
                        help='Initial gate logit for NS hash embeddings.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    # Supervised contrastive auxiliary loss (v22).
    parser.add_argument('--use_supcon', action='store_true', default=False,
                        help='Add a supervised contrastive (SupCon) loss on the '
                             'final representation using the conversion label: '
                             'loss = main + supcon_weight * SupCon.')
    parser.add_argument('--supcon_weight', type=float, default=0.1,
                        help='Weight lambda on the SupCon term (typical 0.05-0.3).')
    parser.add_argument('--supcon_temp', type=float, default=0.1,
                        help='SupCon softmax temperature (typical 0.05-0.2).')
    parser.add_argument('--supcon_proj_dim', type=int, default=0,
                        help='Projection-head output dim for SupCon (0 = d_model).')
    parser.add_argument('--supcon_pos_anchor_only', action='store_true', default=True,
                        help='v23: only the positive class (converted samples) act '
                             'as SupCon anchors. Avoids the ill-posed "pull all '
                             'non-converted together" objective under imbalance.')
    parser.add_argument('--no_supcon_pos_anchor_only', dest='supcon_pos_anchor_only',
                        action='store_false',
                        help='Use all samples as SupCon anchors (v22 behavior).')

    # Query pooling (v25: DIN-MLP target-aware weighted pooling, ported from v17).
    parser.add_argument('--query_pool_mode', type=str, default='mean',
                        choices=['mean', 'target_attn'],
                        help='Pooling that summarizes each sequence into the Q '
                             'generator input. mean = masked mean (v24 behavior); '
                             'target_attn = DIN-MLP target-aware weighted pooling.')

    # word2vec warm-start for high-card sequence ids (v28). Trained in-job;
    # the fine-tuned embeddings live in the single model.pt.
    parser.add_argument('--use_seq_w2v', action='store_true', default=False,
                        help='Init the listed high-card seq id embeddings from an '
                             'in-job word2vec, then fine-tune (replaces hash/direct '
                             'embedding for those ids).')
    parser.add_argument('--seq_w2v_ids', type=str,
                        default='seq_a:38,seq_b:69,seq_c:47,seq_d:23',
                        help='Which seq ids get word2vec init, format domain:fid,...')
    parser.add_argument('--seq_w2v_top_k', type=int, default=2000000,
                        help='Keep at most this many most-frequent ids per feature.')
    parser.add_argument('--seq_w2v_min_count', type=int, default=5,
                        help='word2vec min_count (drop ids rarer than this).')
    parser.add_argument('--seq_w2v_window', type=int, default=5,
                        help='word2vec context window.')
    parser.add_argument('--seq_w2v_epochs', type=int, default=3,
                        help='word2vec training epochs.')
    parser.add_argument('--seq_w2v_workers', type=int, default=8,
                        help='word2vec worker threads.')
    parser.add_argument('--seq_w2v_max_sentences', type=int, default=2000000,
                        help='Cap on sentences (rows) used to train word2vec.')
    parser.add_argument('--seq_w2v_max_seq_len', type=int, default=200,
                        help='Truncate each behavior sentence to this length.')
    parser.add_argument('--seq_w2v_freeze', action='store_true', default=False,
                        help='Freeze the word2vec embeddings (use them as-is, no '
                             'fine-tuning during Hyformer training).')
    parser.add_argument('--use_ns_w2v', action='store_true', default=False,
                        help='Init listed user/item NS int features from in-job '
                             'word2vec (replaces NS hash/direct embedding).')
    parser.add_argument('--ns_w2v_ids', type=str, default='',
                        help='Which NS int fids get word2vec init, format '
                             'user:116,item:123,...')
    parser.add_argument('--ns_w2v_freeze', action='store_true', default=False,
                        help='Freeze NS word2vec embeddings.')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    # ── DDP initialization ──
    rank, local_rank, world_size = setup_ddp()
    ddp_enabled = world_size > 1

    args = parse_args()

    # DDP: override device to local_rank.
    if ddp_enabled:
        args.device = f'cuda:{local_rank}'

    # Create output directories (only rank 0).
    if is_main_process():
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        if args.tf_events_dir:
            Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # DDP barrier: wait for rank 0 to finish creating directories.
    if ddp_enabled:
        dist.barrier()

    # Initialize logger and RNG.
    set_seed(args.seed + rank)  # Different seed per rank for data diversity.

    if is_main_process():
        create_logger(os.path.join(args.log_dir, 'train.log'))
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.info(f"DDP: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    logging.info(f"Args: {vars(args)}")

    writer = None
    if is_main_process() and args.tf_events_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed + rank,
        seq_max_lens=seq_max_lens,
        time_tz_offset_hours=args.time_tz_offset_hours,
        ddp_rank=rank,
        ddp_world_size=world_size,
        split_by_time=args.split_by_time,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- word2vec warm-start for high-card seq ids (v28) ----
    # rank0 trains word2vec once and saves a temp .npz; all ranks load it so
    # every rank builds identical compressed vocabs (DDP-safe). The temp file
    # is intermediate (deleted after load); only the fine-tuned embeddings in
    # model.pt are shipped.
    seq_w2v_slots = None
    w2v_init_data = None
    if args.use_seq_w2v:
        slot_of = {}     # (domain, fid) -> slot index in that domain's side-info
        specs = []       # (domain, prefix, fid)
        for pair in args.seq_w2v_ids.split(','):
            dom, fid = pair.split(':')
            dom, fid = dom.strip(), int(fid)
            side = pcvr_dataset.sideinfo_fids.get(dom, [])
            if fid not in side:
                logging.warning("seq_w2v: fid %d not in %s side-info, skip", fid, dom)
                continue
            slot_of[(dom, fid)] = side.index(fid)
            specs.append((dom, pcvr_dataset._seq_prefix[dom], fid))

        # rank0 trains word2vec once, then saves or reuses an atomically written
        # .npz. Other ranks poll the filesystem instead of entering a long NCCL
        # wait. Once every rank has loaded the file, a fast barrier is safe.
        w2v_cache_npz = build_seq_w2v_cache_path(
            args, schema_path, pcvr_dataset._parquet_files, specs)
        if w2v_cache_npz:
            w2v_npz = w2v_cache_npz
            logging.info("seq_w2v cache path: %s", w2v_npz)
        else:
            w2v_npz = os.path.join(args.ckpt_dir, '_seq_w2v_init.npz')
            logging.info("USER_CACHE_PATH is not set; seq_w2v uses temp init %s", w2v_npz)
        if is_main_process():
            if w2v_cache_npz and os.path.exists(w2v_npz):
                logging.info("seq_w2v cache hit: %s", w2v_npz)
            else:
                from seq_word2vec import build_all_seq_w2v
                res = build_all_seq_w2v(
                    pcvr_dataset._parquet_files, specs,
                    emb_dim=args.emb_dim, top_k=args.seq_w2v_top_k,
                    min_count=args.seq_w2v_min_count, window=args.seq_w2v_window,
                    epochs=args.seq_w2v_epochs, workers=args.seq_w2v_workers,
                    max_sentences=args.seq_w2v_max_sentences,
                    max_seq_len=args.seq_w2v_max_seq_len, seed=args.seed)
                save = {}
                for (dom, fid), (kept, mat) in res.items():
                    save['%s:%d:ids' % (dom, fid)] = kept
                    save['%s:%d:mat' % (dom, fid)] = mat
                save_npz_atomic(w2v_npz, save)
                logging.info("seq_w2v init saved to %s", w2v_npz)
        elif ddp_enabled:
            # Non-rank0: filesystem poll (no NCCL collective -> no watchdog).
            waited = 0
            while not os.path.exists(w2v_npz):
                time.sleep(10)
                waited += 10
                if waited % 300 == 0:
                    logging.info("waiting for rank0 word2vec (%ds)...", waited)
                if waited > 86400:  # 24h hard safety
                    raise TimeoutError("timed out waiting for word2vec init file")

        npz = np.load(w2v_npz)
        w2v_init_data = {}
        seq_w2v_slots = {}
        for (dom, fid), slot in slot_of.items():
            kept = npz['%s:%d:ids' % (dom, fid)]
            mat = npz['%s:%d:mat' % (dom, fid)]
            w2v_init_data[(dom, slot)] = (kept, mat)
            seq_w2v_slots.setdefault(dom, {})[slot] = int(kept.shape[0])
        npz.close()
        # Everyone has loaded now and arrives here within seconds -> this
        # barrier is fast and safe (unlike one spanning the long w2v step).
        if ddp_enabled:
            dist.barrier()
        if is_main_process() and not w2v_cache_npz:
            for _p in (w2v_npz,):
                try:
                    os.remove(_p)
                except OSError:
                    pass
        logging.info("seq_w2v slots: %s", seq_w2v_slots)

    # ---- word2vec warm-start for NS-side user/item int ids ----
    ns_w2v_slots = None
    ns_w2v_init_data = None
    if args.use_ns_w2v:
        from seq_word2vec import build_all_column_w2v

        schema_by_side = {
            'user': pcvr_dataset.user_int_schema,
            'item': pcvr_dataset.item_int_schema,
        }
        prefix_by_side = {
            'user': 'user_int_feats',
            'item': 'item_int_feats',
        }
        idx_of = {}
        specs = []  # (side, fid, column_name)
        for pair in args.ns_w2v_ids.split(','):
            if not pair.strip():
                continue
            side, fid = pair.split(':')
            side, fid = side.strip(), int(fid)
            if side not in schema_by_side:
                logging.warning("ns_w2v: unknown side %s, skip", side)
                continue
            fid_to_idx = {
                f: i for i, (f, _, _) in enumerate(schema_by_side[side].entries)
            }
            if fid not in fid_to_idx:
                logging.warning("ns_w2v: fid %d not in %s_int schema, skip", fid, side)
                continue
            idx_of[(side, fid)] = fid_to_idx[fid]
            specs.append((side, fid, '%s_%d' % (prefix_by_side[side], fid)))

        ns_w2v_cache_npz = build_seq_w2v_cache_path(
            args, schema_path, pcvr_dataset._parquet_files, specs)
        if ns_w2v_cache_npz:
            ns_w2v_npz = ns_w2v_cache_npz.replace('seq_w2v_', 'ns_w2v_')
            logging.info("ns_w2v cache path: %s", ns_w2v_npz)
        else:
            ns_w2v_npz = os.path.join(args.ckpt_dir, '_ns_w2v_init.npz')
            logging.info("USER_CACHE_PATH is not set; ns_w2v uses temp init %s", ns_w2v_npz)
        if is_main_process():
            if ns_w2v_cache_npz and os.path.exists(ns_w2v_npz):
                logging.info("ns_w2v cache hit: %s", ns_w2v_npz)
            else:
                res = build_all_column_w2v(
                    pcvr_dataset._parquet_files, specs,
                    emb_dim=args.emb_dim, top_k=args.seq_w2v_top_k,
                    min_count=args.seq_w2v_min_count, window=args.seq_w2v_window,
                    epochs=args.seq_w2v_epochs, workers=args.seq_w2v_workers,
                    max_sentences=args.seq_w2v_max_sentences,
                    max_seq_len=args.seq_w2v_max_seq_len, seed=args.seed)
                save = {}
                for (side, fid), (kept, mat) in res.items():
                    save['%s:%d:ids' % (side, fid)] = kept
                    save['%s:%d:mat' % (side, fid)] = mat
                save_npz_atomic(ns_w2v_npz, save)
                logging.info("ns_w2v init saved to %s", ns_w2v_npz)
        elif ddp_enabled:
            waited = 0
            while not os.path.exists(ns_w2v_npz):
                time.sleep(10)
                waited += 10
                if waited % 300 == 0:
                    logging.info("waiting for rank0 NS word2vec (%ds)...", waited)
                if waited > 86400:  # 24h hard safety
                    raise TimeoutError("timed out waiting for NS word2vec init file")

        npz = np.load(ns_w2v_npz)
        ns_w2v_init_data = {}
        ns_w2v_slots = {}
        for (side, fid), fid_idx in idx_of.items():
            kept = npz['%s:%d:ids' % (side, fid)]
            mat = npz['%s:%d:mat' % (side, fid)]
            ns_w2v_init_data[(side, fid_idx)] = (kept, mat)
            ns_w2v_slots.setdefault(side, {})[fid_idx] = int(kept.shape[0])
        npz.close()
        if ddp_enabled:
            dist.barrier()
        if is_main_process() and not ns_w2v_cache_npz:
            try:
                os.remove(ns_w2v_npz)
            except OSError:
                pass
        logging.info("ns_w2v slots: %s", ns_w2v_slots)

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "user_dense_feature_specs": pcvr_dataset.user_dense_schema.entries,
        "item_dense_feature_specs": pcvr_dataset.item_dense_schema.entries,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "use_calendar_time_features": args.use_calendar_time_features,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "seq_hash_bucket_size": args.seq_hash_bucket_size,
        "seq_hash_gate_init": args.seq_hash_gate_init,
        "ns_hash_bucket_size": args.ns_hash_bucket_size,
        "ns_hash_gate_init": args.ns_hash_gate_init,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_supcon": args.use_supcon,
        "supcon_proj_dim": args.supcon_proj_dim,
        "query_pool_mode": args.query_pool_mode,
        "seq_w2v_slots": seq_w2v_slots,
        "ns_w2v_slots": ns_w2v_slots,
        "seq_w2v_freeze": args.seq_w2v_freeze,
        "ns_w2v_freeze": args.ns_w2v_freeze,
    }

    model = PCVRHyFormer(**model_args).to(args.device)
    if w2v_init_data is not None:
        # Load word2vec init before DDP wrap; DDP then broadcasts rank0's
        # weights/buffers so all ranks stay in sync.
        model.load_seq_w2v_init(w2v_init_data)
    if ns_w2v_init_data is not None:
        model.load_ns_w2v_init(ns_w2v_init_data)

    # ── DDP wrapping ──
    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
        logging.info(f"Model wrapped with DDP on device cuda:{local_rank}")

    # Log model sizing info.
    raw_model = model.module if ddp_enabled else model
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = raw_model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        use_supcon=args.use_supcon,
        supcon_weight=args.supcon_weight,
        supcon_temp=args.supcon_temp,
        supcon_pos_anchor_only=args.supcon_pos_anchor_only,
        ema_decay=args.ema_decay,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config={**vars(args), 'seq_w2v_slots': seq_w2v_slots,
                      'ns_w2v_slots': ns_w2v_slots},
        save_every_epoch=args.save_every_epoch,
    )

    trainer.train()

    if writer:
        writer.close()

    logging.info("Training complete!")
    cleanup_ddp()


if __name__ == "__main__":
    main()
