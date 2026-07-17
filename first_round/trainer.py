"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping, ModelEMA
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        pair_loss_weight: float = 0.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        dense_stats_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        amp_dtype: str = 'bf16',
        ema_decay: float = 0.0,
        compile_model: bool = False,
        compile_mode: str = 'default',
        compile_eval: bool = False,
        train_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # Dense value scaling stats are copied next to schema.json so infer.py
        # can apply the exact same transform as training.
        self.dense_stats_path: Optional[str] = dense_stats_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.pair_loss_weight: float = pair_loss_weight
        self.ema: Optional[ModelEMA] = None
        if ema_decay > 0:
            # Exclude sparse (embedding) parameters from EMA: high-cardinality
            # embeddings are updated too infrequently per row for EMA to
            # provide meaningful smoothing, and the periodic reinit of sparse
            # params would leave EMA shadow stale (shadow is not reset while
            # param.data is re-initialized with xavier_normal_).
            sparse_ptrs: set = set()
            if hasattr(model, 'get_sparse_params'):
                sparse_ptrs = {p.data_ptr() for p in model.get_sparse_params()}
            ema_exclude_names = {
                name for name, p in model.named_parameters()
                if p.data_ptr() in sparse_ptrs
            }
            self.ema = ModelEMA(model, decay=ema_decay,
                                exclude_names=ema_exclude_names)
            logging.info(f"EMA enabled: decay={ema_decay}, "
                         f"excluded {len(ema_exclude_names)} sparse param tensors, "
                         f"tracking {len(self.ema.shadow)} dense param tensors")

        # SWA: track top-2 best checkpoints for weight averaging.
        # State dicts are saved to disk (not memory) to avoid OOM on large
        # embedding models.  Each entry is (auc, temp_filepath, global_step).
        self._swa_top2: list = []  # sorted descending by AUC
        # Running max validation AUC, used to stop as soon as AUC drops.
        self._swa_best_auc: Optional[float] = None

        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.device_type: str = torch.device(device).type
        self.requested_amp_dtype: str = amp_dtype.lower()
        self.use_amp, self.amp_torch_dtype = self._resolve_amp_config(
            self.requested_amp_dtype)
        self.use_grad_scaler: bool = (
            self.use_amp and self.amp_torch_dtype == torch.float16
        )
        self.scaler: Optional[GradScaler] = (
            GradScaler(enabled=True) if self.use_grad_scaler else None
        )
        self._valid_eval_interval = max(1, eval_every_n_steps) if eval_every_n_steps > 0 else 0

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"pair_loss_weight={pair_loss_weight}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"amp_dtype={self.requested_amp_dtype}, "
                     f"use_amp={self.use_amp}, "
                     f"autocast_dtype={self.amp_torch_dtype}, "
                     f"use_grad_scaler={self.use_grad_scaler}")

        # torch.compile wrappers. self.model stays uncompiled so EMA, the dual
        # optimizers, state_dict, and the predict() entry-point all see the
        # original parameters. The compiled callables are used only for the hot
        # forward paths in _train_step / _evaluate_step. dynamic=True avoids
        # recompiling each time per-domain sequence padding lengths change.
        self.compile_model: bool = bool(compile_model)
        self.compile_mode: str = compile_mode
        self.compile_eval: bool = bool(compile_eval)
        self._forward_compiled = self.model
        # Keep validation/predict eager by default. This avoids first-validation
        # torch.compile graph breaks / long compile stalls while preserving an
        # optional training-forward compile path for explicit benchmarks.
        self._predict_compiled = self.model.predict
        if self.compile_model:
            if not hasattr(torch, 'compile'):
                logging.warning(
                    "torch.compile requested but not available in this PyTorch "
                    "build; running eager.")
                self.compile_model = False
                self.compile_eval = False
            else:
                logging.info(
                    f"Compiling training forward with "
                    f"torch.compile(mode={compile_mode}, dynamic=True)")
                self._forward_compiled = torch.compile(
                    self.model, mode=compile_mode, dynamic=True)
                if self.compile_eval:
                    logging.info(
                        f"Compiling validation predict with "
                        f"torch.compile(mode={compile_mode}, dynamic=True)")
                    self._predict_compiled = torch.compile(
                        self.model.predict, mode=compile_mode, dynamic=True)
                else:
                    logging.info("Validation predict keeps eager mode (--compile_eval not set).")
        else:
            logging.info("torch.compile disabled; training and validation use eager mode.")

    def _resolve_amp_config(
        self, amp_dtype: str
    ) -> Tuple[bool, Optional[torch.dtype]]:
        """Resolve the requested AMP dtype to runtime autocast settings."""
        if amp_dtype == 'fp32':
            return False, None
        if amp_dtype not in {'bf16', 'fp16'}:
            raise ValueError(
                f"amp_dtype must be one of ['bf16', 'fp16', 'fp32'], got {amp_dtype}")
        if self.device_type != 'cuda' or not torch.cuda.is_available():
            logging.info(
                f"AMP dtype {amp_dtype} requested but device={self.device}; "
                "using fp32.")
            return False, None
        if amp_dtype == 'bf16':
            is_bf16_supported = getattr(torch.cuda, 'is_bf16_supported', None)
            if is_bf16_supported is not None and not is_bf16_supported():
                logging.warning(
                    "BF16 AMP requested but the current CUDA device does not "
                    "report BF16 support; using fp32. Use --amp_dtype fp16 "
                    "if FP16 AMP is desired on this GPU.")
                return False, None
            return True, torch.bfloat16
        return True, torch.float16

    def _autocast_context(self) -> Any:
        """Return a CUDA autocast context or a no-op context when AMP is off."""
        if not self.use_amp or self.amp_torch_dtype is None:
            return nullcontext()
        if hasattr(torch, 'autocast'):
            return torch.autocast(
                device_type=self.device_type,
                dtype=self.amp_torch_dtype,
                enabled=True,
            )
        return torch.cuda.amp.autocast(
            enabled=True,
            dtype=self.amp_torch_dtype,
        )

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to four files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``dense_stats.json`` (copied from ``self.dense_stats_path`` when
          set and the file exists): mean/std used for dense value scaling.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        if self.dense_stats_path and os.path.exists(self.dense_stats_path):
            shutil.copy2(self.dense_stats_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    # ---- SWA (Stochastic Weight Averaging) helpers ----

    def _update_swa_top2(
        self, val_auc: float, state_dict: Dict[str, torch.Tensor],
        global_step: int,
    ) -> bool:
        """Keep at most two (auc, filepath, step) entries, sorted desc.

        State dicts are written to temporary files under ``self.save_dir``
        instead of being held in memory, avoiding OOM on models with large
        embedding tables.

        Returns:
            ``True`` iff the validation AUC has *dropped* relative to the best
            AUC seen so far (``val_auc < running_best``). The caller uses this
            to stop training early and proceed straight to fusion.
        """
        # Save the snapshot to a temp file immediately.
        tmp_path = os.path.join(
            self.save_dir, f"_swa_tmp_step{global_step}.pt")
        torch.save(state_dict, tmp_path)
        print(f"[SWA] Saved snapshot to {tmp_path} (AUC={val_auc:.6f})")

        entry = (val_auc, tmp_path, global_step)
        self._swa_top2.append(entry)
        # Sort descending by AUC, keep top-2.
        self._swa_top2.sort(key=lambda x: x[0], reverse=True)
        if len(self._swa_top2) > 2:
            # Remove the evicted entry's temp file.
            _, evicted_path, _ = self._swa_top2.pop()
            if os.path.exists(evicted_path):
                os.remove(evicted_path)
                print(f"[SWA] Evicted {evicted_path}")
        aucs = [f"{e[0]:.6f}(step={e[2]})" for e in self._swa_top2]
        print(f"[SWA] Top-2 so far: {aucs}")
        logging.info(f"SWA top-2 updated: {aucs}")

        # Detect a drop against the running best (computed *before* this round
        # updates the running max).
        dropped = (
            self._swa_best_auc is not None and val_auc < self._swa_best_auc
        )
        if self._swa_best_auc is None or val_auc > self._swa_best_auc:
            self._swa_best_auc = val_auc
        return dropped

    def _save_swa_checkpoints(self) -> None:
        """Save 3 checkpoints: SWA-fused, best, second-best.

        Called at the end of training.  Loads top-2 snapshots from their
        temp files, writes the final directories under ``self.save_dir``
        using the same naming convention as ``_build_step_dir_name`` so
        that the platform can recognise and publish them:
        - ``global_stepN...swa_fused``
        - ``global_stepN...swa_best``
        - ``global_stepN...swa_second``
        """
        print(f"\n[SWA] === Saving final checkpoints ({len(self._swa_top2)} candidates) ===")
        if len(self._swa_top2) < 2:
            msg = (f"SWA requires at least 2 validated checkpoints; "
                   f"only {len(self._swa_top2)} available. Skipping SWA fusion.")
            print(f"[SWA] WARNING: {msg}")
            logging.warning(msg)
            if len(self._swa_top2) == 1:
                best_auc, best_path, best_step = self._swa_top2[0]
                best_sd = torch.load(best_path, map_location="cpu")
                dir_name = self._build_step_dir_name(best_step).replace(
                    ".best_model", "") + ".swa_best"
                best_dir = os.path.join(self.save_dir, dir_name)
                os.makedirs(best_dir, exist_ok=True)
                torch.save(best_sd, os.path.join(best_dir, "model.pt"))
                self._write_sidecar_files(best_dir)
                print(f"[SWA] Saved only-best to {best_dir} "
                      f"(AUC={best_auc:.6f})")
            self._cleanup_swa_tmp()
            return

        best_auc, best_path, best_step = self._swa_top2[0]
        second_auc, second_path, second_step = self._swa_top2[1]

        # Build base name (without .best_model suffix) for consistent naming.
        base_name = self._build_step_dir_name(best_step)

        # Load the two snapshots from disk.
        print(f"[SWA] Loading best (AUC={best_auc:.6f}, step={best_step}) ...")
        best_sd = torch.load(best_path, map_location="cpu")
        print(f"[SWA] Loading second (AUC={second_auc:.6f}, step={second_step}) ...")
        second_sd = torch.load(second_path, map_location="cpu")

        # 1) Save best
        best_dir = os.path.join(self.save_dir, base_name + ".swa_best")
        os.makedirs(best_dir, exist_ok=True)
        torch.save(best_sd, os.path.join(best_dir, "model.pt"))
        self._write_sidecar_files(best_dir)
        print(f"[SWA] Saved swa_best → {best_dir}")
        logging.info(
            f"SWA: saved best checkpoint to {best_dir} "
            f"(AUC={best_auc:.6f}, step={best_step})")

        # 2) Save second-best
        second_base = self._build_step_dir_name(second_step)
        second_dir = os.path.join(self.save_dir, second_base + ".swa_second")
        os.makedirs(second_dir, exist_ok=True)
        torch.save(second_sd, os.path.join(second_dir, "model.pt"))
        self._write_sidecar_files(second_dir)
        print(f"[SWA] Saved swa_second → {second_dir}")
        logging.info(
            f"SWA: saved second-best checkpoint to {second_dir} "
            f"(AUC={second_auc:.6f}, step={second_step})")

        # 3) Fuse (weighted average: best 0.6, second 0.4) and save
        best_w, second_w = 0.6, 0.4
        fused_sd = {}
        for key in best_sd:
            if best_sd[key].is_floating_point():
                fused_sd[key] = best_w * best_sd[key] + second_w * second_sd[key]
            else:
                fused_sd[key] = best_sd[key].clone()

        fused_dir = os.path.join(self.save_dir, base_name + ".swa_fused")
        os.makedirs(fused_dir, exist_ok=True)
        torch.save(fused_sd, os.path.join(fused_dir, "model.pt"))
        self._write_sidecar_files(fused_dir)
        print(f"[SWA] Saved swa_fused → {fused_dir}")
        print(f"[SWA] === Done! 3 checkpoints saved under {self.save_dir} ===\n")
        logging.info(
            f"SWA: saved fused checkpoint to {fused_dir} "
            f"(weighted {best_w}*best + {second_w}*second of "
            f"AUC={best_auc:.6f} step={best_step} "
            f"& AUC={second_auc:.6f} step={second_step})")

        # Free memory and clean up temp files.
        del best_sd, second_sd, fused_sd
        self._cleanup_swa_tmp()

    def _cleanup_swa_tmp(self) -> None:
        """Remove temporary ``_swa_tmp_*.pt`` files."""
        for entry in self._swa_top2:
            path = entry[1]
            if os.path.exists(path):
                os.remove(path)
        # Also glob in case of stale files from interrupted runs.
        import glob as _glob
        for f in _glob.glob(os.path.join(self.save_dir, "_swa_tmp_*.pt")):
            os.remove(f)

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                if self._valid_eval_interval > 0 and total_step % self._valid_eval_interval == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    if self.ema is not None:
                        self.ema.apply_shadow(self.model)
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)
                    # SWA: snapshot current (EMA-applied) weights and check
                    # whether the validation AUC has dropped.
                    auc_dropped = self._update_swa_top2(
                        val_auc, self.model.state_dict(), total_step)
                    if self.ema is not None:
                        self.ema.restore(self.model)

                    # Validation AUC dropped: stop now and fuse straight away.
                    if auc_dropped:
                        logging.info(
                            f"Validation AUC dropped at step {total_step} "
                            f"({val_auc:.6f} < best {self._swa_best_auc:.6f}); "
                            f"stopping early and fusing.")
                        self._save_swa_checkpoints()
                        return

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        self._save_swa_checkpoints()
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            if self.ema is not None:
                self.ema.apply_shadow(self.model)
            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)
            # SWA: snapshot current (EMA-applied) weights and check whether the
            # validation AUC has dropped.
            auc_dropped = self._update_swa_top2(
                val_auc, self.model.state_dict(), total_step)
            if self.ema is not None:
                self.ema.restore(self.model)

            # Validation AUC dropped: stop now and fuse straight away.
            if auc_dropped:
                logging.info(
                    f"Validation AUC dropped at epoch {epoch} "
                    f"({val_auc:.6f} < best {self._swa_best_auc:.6f}); "
                    f"stopping early and fusing.")
                self._save_swa_checkpoints()
                return

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

        # SWA: save the fused, best, and second-best checkpoints
        self._save_swa_checkpoints()

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_time_feats: Dict[str, torch.Tensor] = {}
        seq_day_type_feats: Dict[str, torch.Tensor] = {}
        batch_size = device_batch['user_int_feats'].shape[0]
        time_feats = device_batch.get(
            'time_feats',
            torch.zeros(batch_size, 3, dtype=torch.long, device=self.device))
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_time_feats[domain] = device_batch.get(
                f'{domain}_time_feats',
                torch.zeros(B, L, 3, dtype=torch.long, device=self.device))
            seq_day_type_feats[domain] = device_batch.get(
                f'{domain}_day_type_feats',
                torch.zeros(B, L, 2, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            user_dense_presence_feats=device_batch.get(
                'user_dense_presence_feats',
                torch.ones_like(device_batch['user_dense_feats'])),
            item_dense_presence_feats=device_batch.get(
                'item_dense_presence_feats',
                torch.ones_like(device_batch['item_dense_feats'])),
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            time_feats=time_feats,
            seq_time_feats=seq_time_feats,
            seq_day_type_feats=seq_day_type_feats,
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad(set_to_none=True)
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad(set_to_none=True)

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits = self._forward_compiled(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)
        logits_for_loss = logits.float()
        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(
                logits_for_loss, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits_for_loss, label)

        # Batch-level pairwise AUC loss: softplus(-(pos - neg))
        if self.pair_loss_weight > 0:
            pos_logits = logits_for_loss[label > 0.5]
            neg_logits = logits_for_loss[label <= 0.5]
            if pos_logits.numel() > 0 and neg_logits.numel() > 0:
                diff = pos_logits[:, None] - neg_logits[None, :]
                pair_loss = F.softplus(-diff).mean()
                loss = loss + self.pair_loss_weight * pair_loss

        if self.use_grad_scaler and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.unscale_(self.sparse_optimizer)
        else:
            loss.backward()

        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        if self.use_grad_scaler and self.scaler is not None:
            self.scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.step(self.sparse_optimizer)
            self.scaler.update()
        else:
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        # EMA update
        if self.ema is not None:
            self.ema.update(self.model)

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad(), self._autocast_context():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().float().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self._predict_compiled(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label
