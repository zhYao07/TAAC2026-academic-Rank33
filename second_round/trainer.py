"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Supports single-GPU and multi-GPU DDP training.

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.algorithms.join import Join
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping, DenseEMA, supcon_loss
from model import ModelInput


def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def is_ddp() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


class _nullcontext:
    """Python 3.6 compatible null context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification (supports DDP).

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
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        use_supcon: bool = False,
        supcon_weight: float = 0.1,
        supcon_temp: float = 0.1,
        supcon_pos_anchor_only: bool = True,
        ema_decay: float = 0.0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        save_every_epoch: bool = False,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        # Full-data training: no validation set -> skip eval/early-stopping and
        # save a checkpoint after every epoch instead of best-by-val.
        self.full_train: bool = valid_loader is None
        self.save_every_epoch: bool = bool(save_every_epoch) or self.full_train
        self.writer = writer
        self.schema_path: Optional[str] = schema_path
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Get the raw model (unwrap DDP if needed).
        self.raw_model = model.module if hasattr(model, 'module') else model

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        # Use raw_model's params (DDP wrapper points to the same storage).
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(self.raw_model, 'get_sparse_params'):
            sparse_params = self.raw_model.get_sparse_params()
            dense_params = self.raw_model.get_dense_params()
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
                self.raw_model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.use_supcon: bool = bool(use_supcon)
        self.supcon_weight: float = supcon_weight
        self.supcon_temp: float = supcon_temp
        self.supcon_pos_anchor_only: bool = bool(supcon_pos_anchor_only)
        self._last_sc_loss: float = 0.0
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self._swa_epoch6_state: Optional[Dict[str, torch.Tensor]] = None

        # Dense-only EMA (sparse embeddings excluded by design — see DenseEMA).
        self.ema_decay: float = ema_decay
        self.ema: Optional[DenseEMA] = None
        if ema_decay and ema_decay > 0.0:
            if hasattr(self.raw_model, 'get_dense_params'):
                ema_params = self.raw_model.get_dense_params()
            else:
                ema_params = list(self.raw_model.parameters())
            self.ema = DenseEMA(ema_params, decay=ema_decay)

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"ema_decay={ema_decay}")

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name."""
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``."""
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
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
        """Save model.pt plus sidecar files (only on rank 0)."""
        if not is_main_process():
            return ""
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            # NOTE: the best model.pt is written by EarlyStopping inside
            # _handle_validation_result, which already swaps in EMA weights.
            # This direct save (only used if called with skip_model_file=False
            # outside that path) saves whatever weights are currently live.
            torch.save(self.raw_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _clone_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        """Clone the current state_dict to CPU for checkpoint composition."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.raw_model.state_dict().items()
        }

    def _dense_param_names(self) -> set:
        """Return dense parameter names; non-dense state is kept from epoch 6."""
        if hasattr(self.raw_model, 'get_dense_params'):
            dense_ptrs = {p.data_ptr() for p in self.raw_model.get_dense_params()}
            return {
                name for name, param in self.raw_model.named_parameters()
                if param.data_ptr() in dense_ptrs
            }
        return {name for name, _ in self.raw_model.named_parameters()}

    def _handle_epoch6_epoch7_swa(
        self,
        epoch: int,
        total_step: int,
        saved_state: Dict[str, torch.Tensor],
    ) -> None:
        """Save dense-only average of epochs 6/7, keeping epoch 6 sparse state."""
        if epoch == 6:
            self._swa_epoch6_state = saved_state
            logging.info("Cached epoch 6 checkpoint state for dense-only SWA")
            return
        if epoch != 7 or self._swa_epoch6_state is None:
            return

        swa_state = {
            name: tensor.clone()
            for name, tensor in self._swa_epoch6_state.items()
        }
        averaged = 0
        for name in self._dense_param_names():
            if name not in swa_state or name not in saved_state:
                continue
            if not torch.is_floating_point(swa_state[name]):
                continue
            swa_state[name] = (
                swa_state[name].float().add(saved_state[name].float()).mul_(0.5)
                .to(dtype=swa_state[name].dtype)
            )
            averaged += 1

        dir_name = f"{self._build_step_dir_name(total_step)}.swa=epoch6-epoch7"
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(swa_state, os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(
            f"Saved dense-only SWA checkpoint to {ckpt_dir}/model.pt "
            f"(averaged_dense_tensors={averaged}, sparse_and_buffers=epoch6)")

    def _save_epoch_checkpoint(self, epoch: int, total_step: int) -> str:
        """Save a per-epoch checkpoint (EMA weights) to its own self-contained
        sub-directory. Used in full-data training where there is no validation
        set to pick a 'best' model. Each dir holds model.pt + schema.json +
        train_config.json (+ ns_groups.json), so inference can point at any one.
        """
        if not is_main_process():
            return ""
        # Save the EMA (deploy) weights, mirroring evaluate()'s EMA swap.
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()
        try:
            dir_name = self._build_step_dir_name(total_step)
            ckpt_dir = os.path.join(self.save_dir, dir_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            saved_state = self._clone_state_dict_cpu()
            torch.save(saved_state, os.path.join(ckpt_dir, "model.pt"))
            self._write_sidecar_files(ckpt_dir)
            self._handle_epoch6_epoch7_swa(epoch, total_step, saved_state)
            logging.info(f"[full_train] saved epoch {epoch} checkpoint -> {ckpt_dir}/model.pt")
            return ckpt_dir
        finally:
            if self.ema is not None:
                self.ema.restore()

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories."""
        if not is_main_process():
            return
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in batch to self.device."""
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
        """Persist a new-best checkpoint atomically (only on rank 0).

        EarlyStopping saves ``raw_model.state_dict()`` directly, so to make the
        persisted best model use the EMA (dense) weights we swap the live params
        to the EMA shadow for the duration of this call and restore afterwards.
        Sparse tables are untouched (no EMA on them) and so are saved as-is.
        """
        if not is_main_process():
            return

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()
        try:
            self._handle_validation_result_impl(total_step, val_auc, val_logloss)
        finally:
            if self.ema is not None:
                self.ema.restore()

    def _handle_validation_result_impl(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:

        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            self.early_stopping(val_auc, self.raw_model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.raw_model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def _broadcast_early_stop(self) -> bool:
        """Broadcast rank 0's early_stop flag to all ranks. Returns whether to stop."""
        if not is_ddp():
            return self.early_stopping.early_stop

        flag = torch.tensor(
            [1 if (is_main_process() and self.early_stopping.early_stop) else 0],
            dtype=torch.long, device=self.device
        )
        dist.broadcast(flag, src=0)
        return flag.item() == 1

    def train(self) -> None:
        """Main training loop with DDP support."""
        if is_main_process():
            print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            num_batches = self.train_loader.dataset.estimated_num_batches()
            loss_sum = 0.0
            epoch_t0 = time.time()

            # Join: ensures ranks that finish their data early participate in
            # empty all-reduce calls, preventing DDP deadlock with IterableDataset.
            join_ctx = Join([self.model]) if is_ddp() else _nullcontext()
            with join_ctx:
                for step, batch in enumerate(self.train_loader):
                    loss = self._train_step(batch)
                    total_step += 1
                    loss_sum += loss

                    if self.writer and is_main_process():
                        self.writer.add_scalar('Loss/train', loss, total_step)
                        if self.use_supcon:
                            self.writer.add_scalar('Loss/supcon', self._last_sc_loss, total_step)

                    if is_main_process() and (step + 1) % 100 == 0:
                        avg_loss = loss_sum / (step + 1)
                        elapsed = time.time() - epoch_t0
                        speed = (step + 1) / elapsed
                        eta = (num_batches - step - 1) / speed if speed > 0 else 0
                        sc_str = f" supcon={self._last_sc_loss:.4f}" if self.use_supcon else ""
                        print(f"Epoch {epoch} | step {step+1}/{num_batches} | "
                              f"loss={loss:.4f} avg={avg_loss:.4f}{sc_str} | "
                              f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s | "
                              f"global_step={total_step}")

                    # Step-level validation (only when eval_every_n_steps > 0
                    # and a validation set exists).
                    if (not self.full_train and self.eval_every_n_steps > 0
                            and total_step % self.eval_every_n_steps == 0):
                        logging.info(f"Evaluating at step {total_step}")
                        val_auc, val_logloss = self.evaluate(epoch=epoch)
                        self.model.train()
                        torch.cuda.empty_cache()

                        if is_main_process():
                            logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")
                            if self.writer:
                                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                        self._handle_validation_result(total_step, val_auc, val_logloss)

                        should_stop = self._broadcast_early_stop()
                        if should_stop:
                            logging.info(f"Early stopping at step {total_step}")
                            return

            train_elapsed = time.time() - epoch_t0
            if is_main_process():
                logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / max(num_batches, 1):.4f}, "
                             f"train time: {train_elapsed/60:.1f}min")

            if self.full_train:
                # No validation set: save this epoch's checkpoint, no eval / no
                # early stopping. Saving happens before the cold restart below
                # so the checkpoint reflects this epoch's trained weights.
                if self.save_every_epoch:
                    self._save_epoch_checkpoint(epoch, total_step)
                if is_main_process():
                    print(f"Epoch {epoch} done | train {train_elapsed/60:.1f}min | "
                          f"full-data, checkpoint saved")
            else:
                eval_t0 = time.time()
                val_auc, val_logloss = self.evaluate(epoch=epoch)
                eval_elapsed = time.time() - eval_t0
                self.model.train()
                torch.cuda.empty_cache()

                if is_main_process():
                    print(f"Epoch {epoch} done | train {train_elapsed/60:.1f}min | "
                          f"eval {eval_elapsed/60:.1f}min")
                    logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")
                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                self._handle_validation_result(total_step, val_auc, val_logloss)

                should_stop = self._broadcast_early_stop()
                if should_stop:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

            # Reinitialize high-cardinality sparse params (cold restart).
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.raw_model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                # DDP: reinit uses random init, broadcast from rank 0 to sync all ranks.
                if is_ddp():
                    for p in self.raw_model.parameters():
                        dist.broadcast(p.data, src=0)
                sparse_params = self.raw_model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ModelInput NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_time_feats: Dict[str, torch.Tensor] = {}
        seq_day_type_feats: Dict[str, torch.Tensor] = {}
        B0 = device_batch['user_int_feats'].shape[0]
        time_feats = device_batch.get(
            'time_feats',
            torch.zeros(B0, 3, dtype=torch.long, device=self.device))
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

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(self.device != 'cpu')):
            if self.use_supcon:
                logits, sc_z = self.model(model_input, return_repr=True)
            else:
                logits = self.model(model_input)  # (B, 1)
                sc_z = None
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)

            if self.use_supcon and sc_z is not None:
                # SupCon in fp32 for numerical stability under autocast.
                sc_loss = supcon_loss(
                    sc_z.float(), label.long(), temperature=self.supcon_temp,
                    pos_anchor_only=self.supcon_pos_anchor_only)
                self._last_sc_loss = float(sc_loss.detach())
                loss = loss + self.supcon_weight * sc_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        # EMA tracks only the densely-updated params; update after the step.
        if self.ema is not None:
            self.ema.update()

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation and return (AUC, logloss).

        In DDP mode, gathers results from all ranks and computes metrics on rank 0.
        Other ranks receive (0.0, 0.0).
        """
        if is_main_process():
            logging.info("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        # Evaluate with EMA (dense) weights; restore raw weights in finally so
        # training continues from the un-averaged params. Done on all ranks so
        # DDP collective ops stay in lockstep.
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()
        try:
            return self._evaluate_impl(epoch)
        finally:
            if self.ema is not None:
                self.ema.restore()

    def _evaluate_impl(self, epoch: int) -> Tuple[float, float]:
        num_batches = self.valid_loader.dataset.estimated_num_batches()
        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            eval_t0 = time.time()
            for step, batch in enumerate(self.valid_loader):
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                if is_main_process() and (step + 1) % 100 == 0:
                    elapsed = time.time() - eval_t0
                    speed = (step + 1) / elapsed
                    eta = (num_batches - step - 1) / speed if speed > 0 else 0
                    print(f"Eval | step {step+1}/{num_batches} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # DDP: gather eval results from all ranks to rank 0.
        if is_ddp():
            all_logits, all_labels = self._gather_eval_results(all_logits, all_labels)

        # Only rank 0 computes metrics; other ranks return placeholder.
        if not is_main_process():
            return 0.0, 0.0

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

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

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _gather_eval_results(
        self, local_logits: torch.Tensor, local_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDP: gather eval results from all ranks to rank 0.

        Uses pad + all_gather to handle variable-length data from IterableDataset.
        """
        device = torch.device(self.device)
        world_size = dist.get_world_size()

        # 1. Gather each rank's sample count.
        local_size = torch.tensor([local_logits.shape[0]], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        sizes = [int(s.item()) for s in size_list]
        max_size = max(sizes)

        # 2. Pad to uniform length and all_gather.
        def _pad_and_gather(tensor, max_sz):
            padded = torch.zeros(max_sz, dtype=tensor.dtype, device=device)
            padded[:tensor.shape[0]] = tensor.to(device)
            gathered = [torch.zeros(max_sz, dtype=tensor.dtype, device=device) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
            parts = [g[:sz].cpu() for g, sz in zip(gathered, sizes)]
            return torch.cat(parts, dim=0)

        all_logits = _pad_and_gather(local_logits, max_size)
        all_labels = _pad_and_gather(local_labels, max_size)

        return all_logits, all_labels

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return (logits, labels)."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(self.device != 'cpu')):
            logits, _ = self.raw_model.predict(model_input)
        logits = logits.squeeze(-1)

        return logits, label
