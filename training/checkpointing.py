import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Checkpointer:
    """Manages saving and loading of training checkpoints.

    Saves model weights, optimizer state, scheduler state, and training
    metadata. Automatically rotates old checkpoints to save disk space.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._saved: List[Path] = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LambdaLR],
        step: int,
        loss: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint to disk.

        Returns:
            Path to the saved checkpoint directory.
        """
        ckpt_path = self.checkpoint_dir / f"step_{step:08d}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Model weights (supports DDP-wrapped and torch.compile-wrapped models)
        model_to_save = model.module if hasattr(model, "module") else model
        if hasattr(model_to_save, "_orig_mod"):   # unwrap torch.compile
            model_to_save = model_to_save._orig_mod
        torch.save(model_to_save.state_dict(), ckpt_path / "model.pt")

        # Optimizer + scheduler
        torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")

        # Metadata
        meta = {
            "step":  step,
            "loss":  loss,
            "extra": extra or {},
        }
        with open(ckpt_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._saved.append(ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}  (loss={loss:.4f})")

        # Rotate old checkpoints
        self._rotate()

        # Always keep a symlink to the latest
        latest = self.checkpoint_dir / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(ckpt_path.name)
        except (OSError, NotImplementedError):
            pass  # Windows may not support symlinks without admin rights

        return ckpt_path

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        checkpoint_path: Optional[str] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            checkpoint_path: explicit path; if None loads the latest checkpoint.

        Returns:
            Metadata dict from meta.json.
        """
        if checkpoint_path is None:
            latest = self.checkpoint_dir / "latest"
            if latest.is_symlink():
                ckpt_path = latest.resolve()
            else:
                # Find highest step checkpoint
                candidates = sorted(self.checkpoint_dir.glob("step_*"))
                if not candidates:
                    raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
                ckpt_path = candidates[-1]
        else:
            ckpt_path = Path(checkpoint_path)

        model_to_load = model.module if hasattr(model, "module") else model
        if hasattr(model_to_load, "_orig_mod"):   # unwrap torch.compile
            model_to_load = model_to_load._orig_mod
        state = torch.load(ckpt_path / "model.pt", map_location=map_location)
        model_to_load.load_state_dict(state)

        if optimizer is not None and (ckpt_path / "optimizer.pt").exists():
            opt_state = torch.load(ckpt_path / "optimizer.pt", map_location=map_location)
            optimizer.load_state_dict(opt_state)

        if scheduler is not None and (ckpt_path / "scheduler.pt").exists():
            sch_state = torch.load(ckpt_path / "scheduler.pt", map_location=map_location)
            scheduler.load_state_dict(sch_state)

        with open(ckpt_path / "meta.json", "r") as f:
            meta = json.load(f)

        print(f"Loaded checkpoint from {ckpt_path}  (step={meta['step']}, loss={meta['loss']:.4f})")
        return meta

    def _rotate(self) -> None:
        """Delete old checkpoints beyond keep_last_n."""
        while len(self._saved) > self.keep_last_n:
            old = self._saved.pop(0)
            if old.exists():
                shutil.rmtree(old)
                print(f"Removed old checkpoint: {old}")

    def list_checkpoints(self) -> List[Path]:
        return sorted(self.checkpoint_dir.glob("step_*"))
