"""
training_loop.py
================
Custom training loop for the MiniLM model.

This module is part of the project:
    "A bilingual PT+EN LLM with BPE tokenizer and training loop
     implemented from scratch, with didactic and documented code"

Author  : André Costa
License : MIT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEORETICAL BACKGROUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The training objective
-----------------------
Training an LLM is an optimization problem: we want to find the
weights θ that minimize the average loss over the corpus:

    L(θ) = -1/N Σ log P(t_i | t_1, ..., t_{i-1}; θ)

In other words: maximize the probability the model assigns to the
correct next token given the previous context. This is called
"Language Modeling" or "next-token prediction".

The standard metric is Perplexity (PPL):
    PPL = exp(L)

Intuitively, perplexity measures "how many words the model considers
equally likely at each step". PPL = 10 means the model is, on average,
as uncertain as if it were choosing between 10 equally probable options.

Stochastic Gradient Descent (SGD)
-----------------------------------
Instead of computing the gradient over the entire corpus (infeasible),
we use mini-batches: random samples of B sequences per step.

    θ ← θ - η × ∇_θ L(batch)

where η is the learning rate.

AdamW Optimizer (Loshchilov & Hutter, 2019)
---------------------------------------------
AdamW combines two insights:
    1. Adam: adaptive per-parameter learning rate using first and
       second order gradient moments
    2. Decoupled weight decay: L2 regularization applied directly
       to weights, without interfering with Adam's adaptation

    m_t = β1 × m_{t-1} + (1-β1) × g_t          (1st order moment)
    v_t = β2 × v_{t-1} + (1-β2) × g_t²          (2nd order moment)
    θ_t = θ_{t-1} - η × m̂_t / (√v̂_t + ε) - η × λ × θ_{t-1}

Typical values: β1=0.9, β2=0.95, ε=1e-8, λ=0.1

Cosine Learning Rate Schedule with Warmup
-------------------------------------------
The learning rate is not constant — it varies throughout training:

    Phase 1 — Linear warmup (first ~2% of steps):
        lr grows linearly from 0 to lr_max
        Avoids instability at the start when weights are random

    Phase 2 — Cosine decay:
        lr decays smoothly from lr_max to lr_min
        lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t/T))

    Cosine decay is preferable to linear because:
        - Decays slowly at the start (still much to learn)
        - Decays faster in the middle
        - Stabilizes near the end (fine-grained refinement)

Gradient Clipping
------------------
Limits the gradient norm to a maximum value (typically 1.0):
    if ||g|| > max_norm:
        g ← g × max_norm / ||g||

Prevents "gradient explosion" — situations where the gradient grows
uncontrollably, causing destructive weight updates.
Especially important at the start of training.

Gradient Accumulation
----------------------
Simulates larger batch sizes without increasing VRAM usage:
    - Instead of one step with batch=32, do 4 steps with batch=8
    - Accumulate gradients across the 4 steps (without optimizer.step())
    - Apply the update after the 4th step

    effective_batch_size = batch_size × accumulation_steps

Useful for the RTX 4060 Ti (16GB), where physical batch size is limited.

Mixed Precision Training (bf16)
---------------------------------
Uses bfloat16 (16 bits) instead of float32 to:
    - Reduce VRAM usage by half
    - Speed up computation (bf16 ops are ~2x faster on modern GPUs)

bf16 vs fp16:
    - fp16: range 6×10⁻⁵ to 65504 → risk of overflow/underflow
    - bf16: same range as fp32 → more stable, no grad scaling needed

The RTX 4060 Ti natively supports bf16 — always use it.

References:
    - Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay
      regularization. ICLR 2019.
    - Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient
      descent with warm restarts. ICLR 2017.
    - Micikevicius, P. et al. (2018). Mixed precision training. ICLR 2018.
"""

import os
import math
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project modules
from transformer import MiniLM, ModelConfig
from data_pipeline import CorpusDataset


# ─────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    Training hyperparameters and settings.

    Separating training configuration from model configuration
    allows experimenting with different optimization regimes using
    the same architecture, and vice versa.

    Fields:
        # Paths
        corpus_dir:         Directory of the pre-processed corpus.
        checkpoint_dir:     Where to save checkpoints during training.
        model_config_path:  Path to save/load the model config.

        # Optimization
        lr_max:             Maximum (peak) learning rate.
                            Typical values for LLMs: 3e-4 to 6e-4.
        lr_min:             Minimum learning rate (end of cosine decay).
                            Typically lr_max / 10.
        weight_decay:       Decoupled L2 regularization in AdamW.
        beta1, beta2:       Adam moments. β2=0.95 is more conservative
                            than the default 0.999 — more stable for LLMs.
        grad_clip:          Maximum gradient norm.

        # Batch and accumulation
        batch_size:         Sequences per GPU step.
        accum_steps:        Gradient accumulation steps.
                            effective_batch = batch_size × accum_steps.

        # Schedule
        warmup_steps:       Linear warmup steps.
        max_steps:          Total optimization steps.
                            None = train for 1 full epoch.

        # Logging and checkpoints
        log_interval:       How often (in steps) to log metrics.
        eval_interval:      How often (in steps) to evaluate on val set.
        save_interval:      How often (in steps) to save a checkpoint.
        eval_steps:         How many batches to use for evaluation.

        # Hardware
        dtype:              Data type for mixed precision.
                            "bfloat16" for RTX 4060 Ti (recommended).
        compile_model:      If True, uses torch.compile() for ~20% speedup.
        num_workers:        DataLoader workers for parallel data loading.
    """
    # Paths
    corpus_dir:        str   = "./corpus"
    checkpoint_dir:    str   = "./checkpoints"
    model_config_path: str   = "./model_config.json"

    # Optimization
    lr_max:       float = 3e-4
    lr_min:       float = 3e-5
    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95
    grad_clip:    float = 1.0

    # Batch
    batch_size:   int   = 8       # adjust according to available VRAM
    accum_steps:  int   = 4       # effective_batch = 32

    # Schedule
    warmup_steps: int           = 200
    max_steps:    Optional[int] = None   # None = 1 full epoch

    # Logging
    log_interval:  int = 10
    eval_interval: int = 200
    save_interval: int = 500
    eval_steps:    int = 50

    # Hardware
    dtype:         str  = "bfloat16"
    compile_model: bool = True
    num_workers:   int  = 4

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.batch_size * self.accum_steps

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = cls()
        for key, value in data.items():
            setattr(config, key, value)
        return config


# ─────────────────────────────────────────────────────────────
# Learning Rate Schedule
# ─────────────────────────────────────────────────────────────

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """
    Compute the learning rate for the current step.

    Implements the standard LLM schedule:
        - Linear warmup from 0 → lr_max over the first `warmup_steps`
        - Cosine decay from lr_max → lr_min until `max_steps`

    Cosine decay is derived from the work of Loshchilov & Hutter (2017)
    on SGDR (Stochastic Gradient Descent with Restarts).
    Here we use only half a cycle (no restarts).

    Args:
        step:         Current optimization step (starts at 0).
        warmup_steps: Duration of the linear warmup.
        max_steps:    Total training steps.
        lr_max:       Maximum learning rate (warmup peak).
        lr_min:       Minimum learning rate (cosine end).

    Returns:
        Learning rate for the current step.

    Example curve (warmup=100, max=1000, lr_max=3e-4, lr_min=3e-5):
        step=0:    lr = 0.0
        step=50:   lr = 1.5e-4  (midpoint of warmup)
        step=100:  lr = 3e-4    (peak)
        step=550:  lr = 1.65e-4 (midpoint of cosine)
        step=1000: lr = 3e-5    (end)
    """
    # Phase 1: linear warmup
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps

    # Beyond max_steps: hold lr_min
    if step >= max_steps:
        return lr_min

    # Phase 2: cosine decay
    # Normalize progress after warmup to [0, 1]
    progress = (step - warmup_steps) / (max_steps - warmup_steps)

    # Half-cosine decay formula
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_min + cosine_decay * (lr_max - lr_min)


# ─────────────────────────────────────────────────────────────
# Metrics and logging
# ─────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Track and record training metrics.

    Maintains a full history of loss and perplexity for
    post-training analysis and learning curve generation.

    Perplexity (PPL) is the main metric for LLMs:
        PPL = exp(cross_entropy_loss)

    Interpretation:
        PPL = 1:    perfect model (impossible in practice)
        PPL = 10:   good for small models on general text
        PPL = 50:   reasonable for very small models
        PPL = 100+: model still learning / difficult corpus
    """

    def __init__(self, log_dir: str):
        """
        Initialize the tracker and configure the logger.

        Args:
            log_dir: Directory where logs and metrics will be saved.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # Full history for post-training analysis
        self.history: list[dict] = []

        # Accumulators for moving average
        self._loss_accum  = 0.0
        self._accum_count = 0

        # Configure logger to write to both file and console
        self.logger = logging.getLogger("MiniLM")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def update(self, loss: float) -> None:
        """Accumulate loss for average computation."""
        self._loss_accum  += loss
        self._accum_count += 1

    def log_step(
        self,
        step: int,
        lr: float,
        tokens_per_sec: float,
        split: str = "train",
    ) -> dict:
        """
        Record metrics for the current step.

        Args:
            step:           Current step.
            lr:             Current learning rate.
            tokens_per_sec: Token throughput per second.
            split:          "train" or "val".

        Returns:
            Dictionary with the recorded metrics.
        """
        avg_loss = self._loss_accum / max(self._accum_count, 1)
        ppl      = math.exp(min(avg_loss, 20))  # clamp to avoid overflow

        metrics = {
            "step":           step,
            "split":          split,
            "loss":           round(avg_loss, 4),
            "perplexity":     round(ppl, 2),
            "lr":             f"{lr:.2e}",
            "tokens_per_sec": int(tokens_per_sec),
        }

        self.history.append(metrics)

        # Format log line
        self.logger.info(
            f"step {step:>6} | {split:<5} | "
            f"loss {avg_loss:.4f} | ppl {ppl:.2f} | "
            f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s"
        )

        # Reset accumulators
        self._loss_accum  = 0.0
        self._accum_count = 0

        return metrics

    def save_history(self) -> None:
        """Save the full history to JSON."""
        path = os.path.join(self.log_dir, "metrics_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model: MiniLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: TrainingConfig,
    model_config: ModelConfig,
    is_best: bool = False,
) -> None:
    """
    Save a full training state checkpoint.

    A checkpoint includes everything needed to resume training
    exactly where it left off:
        - Model weights (state_dict)
        - Optimizer state (accumulated Adam moments)
        - Current step and best loss (for comparison)
        - Model and training configurations

    Checkpoint strategy:
        - Saves a periodic checkpoint every `save_interval` steps
        - Keeps only the 3 most recent checkpoints (saves disk space)
        - Separately saves the "best checkpoint" (lowest val loss)

    Args:
        model:        Model to save.
        optimizer:    Optimizer with its internal state.
        step:         Current step.
        loss:         Current validation loss.
        config:       Training configuration.
        model_config: Architecture configuration.
        is_best:      If True, also saves as "best_model.pt".
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint = {
        "step":         step,
        "loss":         loss,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "model_config": model_config.__dict__,
        "train_config": {k: v for k, v in config.__dict__.items()
                         if not callable(v)},
    }

    # Periodic checkpoint
    ckpt_path = os.path.join(config.checkpoint_dir, f"ckpt_step_{step:07d}.pt")
    torch.save(checkpoint, ckpt_path)

    # Keep only the 3 most recent
    ckpts = sorted(Path(config.checkpoint_dir).glob("ckpt_step_*.pt"))
    for old_ckpt in ckpts[:-3]:
        old_ckpt.unlink()

    # Save as best model if applicable
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"  → New best model saved (loss={loss:.4f})")


def load_checkpoint(
    path: str,
    model: MiniLM,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Load a saved checkpoint.

    Args:
        path:      Path to the checkpoint .pt file.
        model:     Model to load weights into.
        optimizer: Optimizer to load state into (optional).

    Returns:
        Dictionary with checkpoint metadata (step, loss, configs).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optim_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state"])

    print(f"Checkpoint loaded: step={checkpoint['step']}, "
          f"loss={checkpoint['loss']:.4f}")

    return checkpoint


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: MiniLM,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    eval_steps: int,
) -> float:
    """
    Evaluate the model on the validation set.

    Disables gradient computation (@torch.no_grad) to save memory
    and speed up evaluation — during evaluation we only need the
    forward pass, not the backward pass.

    Loss is computed over `eval_steps` random batches from the val
    set, which is sufficient for a reliable estimate without running
    the full val set (which would be slow).

    Args:
        model:       Model to evaluate.
        val_loader:  DataLoader for the validation set.
        device:      Device (cuda/cpu).
        dtype:       Data type for autocast.
        eval_steps:  How many batches to evaluate.

    Returns:
        Average validation loss.
    """
    model.eval()

    total_loss = 0.0
    steps_done = 0

    for batch in val_loader:
        if steps_done >= eval_steps:
            break

        # Prepare input and targets
        # input_ids: all tokens except the last
        # targets:   all tokens except the first (shift of 1)
        input_ids = batch[:, :-1].to(device)
        targets   = batch[:, 1:].to(device)

        # Forward pass with autocast
        with torch.autocast(device_type=device.type, dtype=dtype):
            _, loss = model(input_ids, targets)

        total_loss += loss.item()
        steps_done += 1

    model.train()
    return total_loss / max(steps_done, 1)


# ─────────────────────────────────────────────────────────────
# Trainer — main class
# ─────────────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates the full training of MiniLM.

    Responsibilities:
        - Set up device, dtype and compilation
        - Initialize model, optimizer and LR schedule
        - Run the training loop with gradient accumulation
        - Periodically evaluate on the val set
        - Save checkpoints and metrics
        - Resume training from a checkpoint

    Basic usage:
        >>> model_config = ModelConfig()
        >>> train_config = TrainingConfig()
        >>> trainer = Trainer(model_config, train_config)
        >>> trainer.train()

    Resuming training:
        >>> trainer = Trainer(model_config, train_config)
        >>> trainer.train(resume_from="./checkpoints/ckpt_step_0005000.pt")
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig):
        """
        Initialize the Trainer.

        Args:
            model_config: Model architecture configuration.
            train_config: Training configuration.
        """
        self.model_config = model_config
        self.config       = train_config

        # ── Device ────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # ── Data type for mixed precision ──────────────────────────────────
        # bf16 for RTX 4060 Ti (Ampere+), fp16 for older GPUs
        if train_config.dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            print("  Mixed precision: bfloat16 ✓")
        elif train_config.dtype == "float16":
            self.dtype = torch.float16
            print("  Mixed precision: float16 ✓")
        else:
            self.dtype = torch.float32
            print("  Mixed precision: disabled (float32)")

        # ── Model ──────────────────────────────────────────────────────────
        self.model = MiniLM(model_config).to(self.device)
        print(f"\nModel: {self.model.count_parameters()['total'] / 1e6:.1f}M parameters")

        # torch.compile() — JIT compilation for ~20% speedup
        # Requires PyTorch 2.0+ and may take a few minutes the first time
        if train_config.compile_model and hasattr(torch, "compile"):
            print("  Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
            print("  torch.compile() ✓")

        # ── Optimizer ──────────────────────────────────────────────────────
        # Weight decay should NOT be applied to:
        #   - Embeddings (weight decay collapses them)
        #   - Bias terms
        #   - Normalization parameters (RMSNorm.weight)
        decay_params    = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params,    "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=train_config.lr_max,
            betas=(train_config.beta1, train_config.beta2),
            eps=1e-8,
            fused=True if self.device.type == "cuda" else False,
            # fused=True: CUDA fused implementation, ~10% faster
        )

        # ── DataLoaders ────────────────────────────────────────────────────
        train_dataset = CorpusDataset(
            os.path.join(train_config.corpus_dir, "train")
        )
        val_dataset = CorpusDataset(
            os.path.join(train_config.corpus_dir, "val")
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.num_workers,
            pin_memory=True,    # speeds up CPU→GPU transfer
            drop_last=True,     # discard incomplete batch at the end
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=True,
        )

        # ── Max steps ──────────────────────────────────────────────────────
        if train_config.max_steps is None:
            # 1 epoch = iterate through the full dataset once
            self.max_steps = len(self.train_loader) // train_config.accum_steps
        else:
            self.max_steps = train_config.max_steps

        print(f"  Max steps: {self.max_steps:,}")
        print(f"  Effective batch size: {train_config.effective_batch_size}")
        print(f"  Steps per epoch: {len(self.train_loader) // train_config.accum_steps:,}")

        # ── Metrics ────────────────────────────────────────────────────────
        self.metrics = MetricsTracker(train_config.checkpoint_dir)

        # ── Internal state ─────────────────────────────────────────────────
        self.step      = 0
        self.best_loss = float("inf")

    def _set_lr(self, step: int) -> float:
        """
        Update the learning rate for all optimizer parameter groups.

        Args:
            step: Current step.

        Returns:
            Computed learning rate.
        """
        lr = get_lr(
            step=step,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.max_steps,
            lr_max=self.config.lr_max,
            lr_min=self.config.lr_min,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Run the full training loop.

        Main loop:
            For each batch from train_loader:
                1. Forward pass → loss
                2. loss /= accum_steps (scale for accumulation)
                3. Backward pass (accumulate gradients)
                4. Every accum_steps:
                    a. Gradient clipping
                    b. Update weights (optimizer.step)
                    c. Zero gradients (optimizer.zero_grad)
                5. Log metrics periodically
                6. Evaluate on val set periodically
                7. Save checkpoint periodically

        Args:
            resume_from: Path to a checkpoint to resume from (optional).
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            ckpt = load_checkpoint(resume_from, self.model, self.optimizer)
            self.step = ckpt["step"]
            self.best_loss = ckpt.get("loss", float("inf"))
            print(f"Resuming from step {self.step}")

        self.model.train()
        self.metrics.logger.info("=" * 60)
        self.metrics.logger.info("Training started")
        self.metrics.logger.info(
            f"max_steps={self.max_steps} | "
            f"batch={self.config.batch_size} | "
            f"accum={self.config.accum_steps} | "
            f"effective_batch={self.config.effective_batch_size}"
        )
        self.metrics.logger.info("=" * 60)

        # Time tracking for throughput computation
        t_start     = time.time()
        tokens_seen = 0

        # Infinite iterator over the dataset
        # (needed since max_steps may span more than 1 epoch)
        def infinite_loader():
            while True:
                for batch in self.train_loader:
                    yield batch

        loader_iter      = infinite_loader()
        accumulated_loss = 0.0

        while self.step < self.max_steps:

            # ── Update learning rate ───────────────────────────────────────
            lr = self._set_lr(self.step)

            # ── Gradient Accumulation Loop ─────────────────────────────────
            # Accumulate gradients over `accum_steps` micro-batches
            # before applying the weight update
            self.optimizer.zero_grad(set_to_none=True)
            # set_to_none=True frees memory instead of zeroing — more efficient

            for _ in range(self.config.accum_steps):
                batch = next(loader_iter)

                # Prepare input and targets (shift of 1 token)
                input_ids = batch[:, :-1].to(self.device, non_blocking=True)
                targets   = batch[:, 1:].to(self.device, non_blocking=True)

                tokens_seen += input_ids.numel()

                # Forward with autocast (mixed precision)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                ):
                    _, loss = self.model(input_ids, targets)

                # Scale the loss by the number of micro-steps so that
                # the accumulated gradient is equivalent to the gradient
                # of a batch of size effective_batch
                loss = loss / self.config.accum_steps
                accumulated_loss += loss.item()

                # Backward: accumulate gradients (do not zero yet)
                loss.backward()

            # ── Weight update ──────────────────────────────────────────────

            # Gradient clipping: prevents gradient explosion
            # Returns the norm before clipping (useful for monitoring)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )

            # Optimization step
            self.optimizer.step()

            self.step += 1

            # ── Logging ────────────────────────────────────────────────────
            self.metrics.update(accumulated_loss)
            accumulated_loss = 0.0

            if self.step % self.config.log_interval == 0:
                elapsed     = time.time() - t_start
                tok_per_sec = tokens_seen / elapsed
                lr_now      = self.optimizer.param_groups[0]["lr"]

                self.metrics.log_step(
                    step=self.step,
                    lr=lr_now,
                    tokens_per_sec=tok_per_sec,
                    split="train",
                )

                # Reset throughput counters
                tokens_seen = 0
                t_start     = time.time()

            # ── Evaluation ─────────────────────────────────────────────────
            if self.step % self.config.eval_interval == 0:
                val_loss = evaluate(
                    model=self.model,
                    val_loader=self.val_loader,
                    device=self.device,
                    dtype=self.dtype,
                    eval_steps=self.config.eval_steps,
                )

                self.metrics._loss_accum  = val_loss
                self.metrics._accum_count = 1
                self.metrics.log_step(
                    step=self.step,
                    lr=self.optimizer.param_groups[0]["lr"],
                    tokens_per_sec=0,
                    split="val",
                )

                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=self.step,
                    loss=val_loss,
                    config=self.config,
                    model_config=self.model_config,
                    is_best=is_best,
                )

            # ── Periodic checkpoint ────────────────────────────────────────
            elif self.step % self.config.save_interval == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=self.step,
                    loss=self.best_loss,
                    config=self.config,
                    model_config=self.model_config,
                    is_best=False,
                )

        # ── End of training ────────────────────────────────────────────────
        self.metrics.logger.info("=" * 60)
        self.metrics.logger.info(
            f"Training complete. "
            f"Best val loss: {self.best_loss:.4f} | "
            f"PPL: {math.exp(self.best_loss):.2f}"
        )
        self.metrics.logger.info("=" * 60)
        self.metrics.save_history()

        print(f"\nBest model saved to: "
              f"{os.path.join(self.config.checkpoint_dir, 'best_model.pt')}")


# ─────────────────────────────────────────────────────────────
# HuggingFace export
# ─────────────────────────────────────────────────────────────

def export_to_huggingface(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_path: str,
) -> None:
    """
    Export the trained model to HuggingFace format.

    Saves the model in a format compatible with AutoModel.from_pretrained(),
    allowing anyone to load the model with:
        model = AutoModel.from_pretrained("your-username/your-model")

    The process:
        1. Load the trained checkpoint
        2. Save weights in safetensors (safe and efficient format)
        3. Create config.json in HuggingFace format
        4. Copy tokenizer files
        5. Create the model card (README.md)

    After this step, use the HuggingFace CLI to publish:
        huggingface-cli upload your-username/minilm ./hf_export

    Args:
        checkpoint_path: Path to best_model.pt.
        output_dir:      Output directory for HF files.
        tokenizer_path:  Directory with BPE tokenizer files.
    """
    import shutil

    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting to HuggingFace format in '{output_dir}'...")

    # Load checkpoint
    ckpt        = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_cfg_dict = ckpt["model_config"]
    # d_head is derived automatically in ModelConfig.__post_init__
    # and must not be passed as a constructor argument
    model_cfg_dict.pop("d_head", None)
    model_config   = ModelConfig(**model_cfg_dict)

    # Instantiate and load weights
    model = MiniLM(model_config)

    # If the model was trained with torch.compile(), the state_dict keys
    # will have a '_orig_mod.' prefix — strip it before loading
    state_dict = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Save weights in safetensors (safer than .bin)
    # Note: weight tying means lm_head.weight and token_emb.weight share
    # the same tensor in memory. safetensors does not allow shared tensors,
    # so we save lm_head.weight as an independent copy.
    try:
        from safetensors.torch import save_file
        tensors = {}
        for k, v in model.state_dict().items():
            # Skip complex tensors (e.g. freqs_complex from RoPE) —
            # safetensors does not support complex dtypes.
            # These buffers are recomputed automatically on model load.
            if v.is_complex():
                continue
            tensors[k] = v.clone()   # clone breaks shared memory references
        save_file(tensors, os.path.join(output_dir, "model.safetensors"))
        print("  Weights saved to model.safetensors")
    except ImportError:
        # Fallback to pytorch_model.bin — supports complex tensors
        state_dict = {k: v for k, v in model.state_dict().items()
                      if not v.is_complex()}
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        print("  Weights saved to pytorch_model.bin")
        print("  (install safetensors for the recommended format: pip install safetensors)")

    # Save config.json in HuggingFace format
    hf_config = {
        "model_type":              "minilm",
        "architectures":           ["MiniLM"],
        "vocab_size":              model_config.vocab_size,
        "hidden_size":             model_config.d_model,
        "num_hidden_layers":       model_config.n_layers,
        "num_attention_heads":     model_config.n_heads,
        "intermediate_size":       model_config.d_ff,
        "max_position_embeddings": model_config.seq_len,
        "hidden_dropout_prob":     model_config.dropout,
        "torch_dtype":             "bfloat16",
        "transformers_version":    "4.0.0",
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    print("  config.json saved")

    # Copy tokenizer files
    for fname in ["tokenizer.json", "vocab.json"]:
        src = os.path.join(tokenizer_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fname))
    print("  Tokenizer files copied")

    # Create model card (README.md)
    params_m = model_config.n_params / 1e6
    readme = f"""---
language:
- pt
- en
license: mit
tags:
- language-model
- bilingual
- portuguese
- english
- from-scratch
---

# MiniLM — Bilingual PT+EN Language Model

A decoder-only Transformer language model trained from scratch,
including a BPE tokenizer and training loop implemented without
high-level frameworks.

## Specifications

| Attribute            | Value                  |
|----------------------|------------------------|
| Parameters           | {params_m:.0f}M               |
| Architecture         | Transformer Decoder-only |
| Normalization        | RMSNorm                |
| Positional Encoding  | RoPE                   |
| FFN Activation       | SwiGLU                 |
| Vocabulary           | {model_config.vocab_size:,} tokens (BPE) |
| Max context          | {model_config.seq_len} tokens          |
| Languages            | Brazilian Portuguese + English |

## Training corpus

- **TinyStories** (EN): short synthetic stories ~60%
- **CulturaX PT** (PT-BR): curated Portuguese web ~40%

## How to use

```python
from bpe_tokenizer import BPETokenizer
from transformer import MiniLM, ModelConfig
import torch, json

tokenizer = BPETokenizer.load("./")

with open("config.json") as f:
    cfg = json.load(f)

model_config = ModelConfig(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["hidden_size"],
    n_layers=cfg["num_hidden_layers"],
    n_heads=cfg["num_attention_heads"],
    d_ff=cfg["intermediate_size"],
    seq_len=cfg["max_position_embeddings"],
)
model = MiniLM(model_config)
model.load_state_dict(torch.load("pytorch_model.bin", map_location="cpu"))
model.eval()

prompt = "Once upon a time"
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
print(tokenizer.decode(output[0].tolist()))
```

## Development

All training code is available in the repository:
- `bpe_tokenizer.py` — BPE tokenizer from scratch
- `data_pipeline.py` — Corpus preparation pipeline
- `transformer.py`   — Model architecture
- `training_loop.py` — Custom training loop

## Citation

```
@misc{{minilm2025,
  title={{MiniLM: A bilingual PT+EN language model built from scratch}},
  author={{André Costa}},
  year={{2026}},
  url={{https://huggingface.co/AndreCosta/minilm}}
}}
```
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print("  README.md (model card) created")

    print(f"\nExport complete!")
    print(f"To publish on HuggingFace:")
    print(f"  huggingface-cli login")
    print(f"  huggingface-cli upload [your-username]/minilm {output_dir}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniLM Training")
    parser.add_argument("--mode", choices=["train", "export"],
                        default="train", help="Execution mode")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to export (export mode)")
    parser.add_argument("--output-dir", type=str, default="./hf_export",
                        help="Output directory for HF export")
    parser.add_argument("--tokenizer-path", type=str, default="./tokenizer",
                        help="Path to the BPE tokenizer")
    parser.add_argument("--small", action="store_true",
                        help="Use Tiny config (~15M params) for quick tests")
    args = parser.parse_args()

    if args.mode == "train":
        # Model configuration
        if args.small:
            print("Using Tiny configuration (~15M params) for quick test")
            model_config = ModelConfig(
                vocab_size=16384,
                seq_len=512,   # must match the seq_len used in data_pipeline.py
                d_model=256,
                n_heads=4,
                n_layers=4,
                d_ff=768,
                dropout=0.1,
            )
            train_config = TrainingConfig(
                batch_size=4,
                accum_steps=2,
                max_steps=100,
                log_interval=10,
                eval_interval=50,
                save_interval=50,
            )
        else:
            model_config = ModelConfig()    # Small (~85M) by default
            train_config = TrainingConfig()

        print("\nModel configuration:")
        print(f"  {model_config.n_params / 1e6:.1f}M parameters")

        trainer = Trainer(model_config, train_config)
        trainer.train(resume_from=args.resume)

    elif args.mode == "export":
        if args.checkpoint is None:
            args.checkpoint = "./checkpoints/best_model.pt"
        export_to_huggingface(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            tokenizer_path=args.tokenizer_path,
        )
