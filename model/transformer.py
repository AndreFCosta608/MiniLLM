"""
transformer.py
==============
Transformer Decoder-only architecture implemented from scratch in PyTorch.

This module is part of the project:
    "A bilingual PT+EN LLM with BPE tokenizer and training loop
     implemented from scratch, with didactic and documented code"

Author  : André Costa
License : MIT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEORETICAL BACKGROUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The Transformer architecture (Vaswani et al., 2017)
-------------------------------------------------
The Transformer originally emerged as an encoder-decoder model for
machine translation. For generative language models, we use only
the decoder half — called "decoder-only" or "causal LM".

This is the architecture used by GPT-2, GPT-3, GPT-4, LLaMA, Mistral,
and virtually all modern LLMs.

Why decoder-only for text generation?
--------------------------------------------
The decoder-only uses causal attention (also called masked attention):
each token can only "see" previous tokens, never future ones.
This allows training the model to predict the next token — the standard
pre-training objective (Language Modeling or LM loss).

    Entrada : [t1, t2, t3, t4]
    Saída   : [t2, t3, t4, t5]   ← cada posição prevê o próximo token

Overview of the implemented architecture
-----------------------------------------
Our implementation incorporates modern improvements over the original
2017 Transformer:

    1. RMSNorm (Zhang & Sennrich, 2019) instead of LayerNorm
       → More efficient: no mean computation, normalizes variance only

    2. RoPE — Rotary Position Embedding (Su et al., 2021) instead of
       absolute positional embeddings
       → Better generalization to sequences longer than those seen in training

    3. SwiGLU (Shazeer, 2020) instead of FFN with ReLU
       → Gated activation learns to "filter" information adaptively

    4. Pre-norm (norm before attention/FFN) instead of post-norm
       → More stable training, healthier gradients

These are exactly the choices made by LLaMA (Touvron et al., 2023),
which have become the industry standard.

Data flow through the model:
    tokens (B, T)
        ↓  nn.Embedding
    x (B, T, d_model)
        ↓  N × TransformerBlock
    x (B, T, d_model)
        ↓  RMSNorm final
    x (B, T, d_model)
        ↓  Linear (lm_head)
    logits (B, T, vocab_size)

    where B = batch size, T = seq_len, d_model = model dimension

Referências:
    - Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.
    - Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization.
    - Su, J. et al. (2021). RoFormer: Enhanced transformer with rotary
      position embedding. arXiv:2104.09864.
    - Shazeer, N. (2020). GLU variants improve transformer. arXiv:2002.05202.
    - Touvron, H. et al. (2023). LLaMA: Open and efficient foundation
      language models. arXiv:2302.13971.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Model configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Model architecture hyperparameters.

    Centralizing configuration in a dataclass allows:
        - Saving and loading the architecture alongside weights
        - Reproducing experiments exactly
        - Varying model sizes without changing the code

    Nomenclature follows literature conventions:
        d_model  = embedding space dimension (also called
                   "hidden size" or "model dimension")
        n_heads  = number of attention heads
        n_layers = number of stacked Transformer blocks
        d_ff     = internal FFN (feed-forward network) dimension,
                   typically 4 × d_model (original) or 8/3 × d_model (SwiGLU)

    Pre-defined configurations (for reference):
        Tiny  (~15M):  d_model=256,  n_heads=4,  n_layers=4,  d_ff=1024
        Small (~85M):  d_model=512,  n_heads=8,  n_layers=8,  d_ff=2048
        Base  (~310M): d_model=768,  n_heads=12, n_layers=12, d_ff=3072
    """
    # Vocabulary and sequence
    vocab_size: int   = 16384   # must match vocab_size of BPETokenizer
    seq_len: int      = 512     # maximum sequence length

    # Model dimensions
    d_model: int      = 512     # embedding dimension
    n_heads: int      = 8       # number of attention heads
    n_layers: int     = 8       # number of Transformer blocks
    d_ff: int         = 1536    # FFN dimension (≈ 3 × d_model for SwiGLU)

    # Regularization
    dropout: float    = 0.1     # dropout applied in attention and FFN

    # Precision
    use_flash: bool   = True    # use Flash Attention if available (PyTorch 2+)

    def __post_init__(self):
        """Validate hyperparameter consistency."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"n_heads ({self.n_heads})"
        )
        # Dimension per attention head
        self.d_head = self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        """
        Estimate the number of model parameters.

        Useful for checking whether the model fits in available VRAM before
        instantiation. The estimate is approximate (ignores bias and buffers).

        Main components:
            - Embedding: vocab_size × d_model
            - Per block: attention (4 × d_model²) + FFN (3 × d_model × d_ff)
            - LM head: d_model × vocab_size (usually tied with embedding)
        """
        embed   = self.vocab_size * self.d_model
        attn    = self.n_layers * 4 * (self.d_model ** 2)
        ffn     = self.n_layers * 3 * self.d_model * self.d_ff
        lm_head = self.d_model * self.vocab_size
        return embed + attn + ffn + lm_head


# ─────────────────────────────────────────────────────────────
# RMSNorm — Root Mean Square Layer Normalization
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    The original LayerNorm normalizes by mean and standard deviation:
        LayerNorm(x) = (x - μ) / (σ + ε) * γ + β

    RMSNorm simplifies: does not subtract the mean (μ = 0 assumed),
    normalizes only by RMS (root mean square):
        RMSNorm(x) = x / RMS(x) * γ
        RMS(x) = sqrt(mean(x²) + ε)

    Advantages:
        - ~15% faster than LayerNorm (no mean computation)
        - No β (bias) parameter, slightly reducing parameter count
        - Same empirical quality in LLMs (used in LLaMA, Mistral, etc.)

    Args:
        d_model: Dimension of the vector to normalize.
        eps:     Numerical stability constant (avoids division by zero).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # γ (gamma): learnable scale parameter, initialized to 1
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to tensor x.

        Args:
            x: Tensor of shape (..., d_model).

        Returns:
            Normalized tensor of same shape as x.
        """
        # Compute RMS along the last dimension (d_model)
        # x.float() ensures numerical precision even with bf16/fp16
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # Normalize and restore original dtype
        x_norm = (x.float() / rms).to(x.dtype)

        # Apply scale parameter γ
        return x_norm * self.weight


# ─────────────────────────────────────────────────────────────
# RoPE — Rotary Position Embedding
# ─────────────────────────────────────────────────────────────

def precompute_rope_freqs(d_head: int, seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    Pre-compute complex frequencies for RoPE.

    RoPE (Su et al., 2021) encodes position by rotating query and key
    vectors in the complex space. The rotation at position m uses
    angle θ_i = m / base^(2i/d), where i indexes the dimension pair.

    Geometric intuition:
        - Each pair of dimensions (2i, 2i+1) forms a 2D plane
        - At position m, we rotate in that plane by m × θ_i
        - The dot product q·k preserves only the position difference (m-n)
        - This gives relative position attention automatically

    Advantages over absolute embeddings:
        - Generalization to seq_len > training seq_len (extrapolation)
        - No extra parameters
        - Attention is naturally sensitive to relative distance

    Args:
        d_head:  Dimension of each attention head.
        seq_len: Maximum sequence length.
        base:    Frequency base (10000 is the RoPE original default).

    Returns:
        Complex tensor of shape (seq_len, d_head // 2) with the frequencies.
    """
    # θ_i = 1 / base^(2i / d_head), for i = 0, 1, ..., d_head/2 - 1
    theta = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))

    # Positions: 0, 1, 2, ..., seq_len-1
    positions = torch.arange(seq_len).float()

    # Outer product: freqs[m, i] = m × θ_i
    # Shape: (seq_len, d_head // 2)
    freqs = torch.outer(positions, theta)

    # Convert to complex form: e^(i × freqs) = cos(freqs) + i×sin(freqs)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex


def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to a query or key tensor.

    Application works in 3 steps:
        1. Interpret consecutive dimension pairs as complex numbers
        2. Multiply by the rotation factor e^(i × m × θ)
        3. Convert back to real tensor

    Args:
        x:             Tensor of shape (B, T, n_heads, d_head).
        freqs_complex: Pre-computed frequencies of shape (T, d_head // 2).

    Returns:
        Rotated tensor of same shape as x.
    """
    B, T, H, D = x.shape

    # Group dimension pairs: (..., d_head) → (..., d_head//2, 2)
    # and interpret as complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))

    # Adjust freqs_complex shape for broadcast: (1, T, 1, d_head//2)
    freqs = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Rotate: complex multiplication applies the rotation
    x_rotated = x_complex * freqs

    # Convert back to real: (B, T, H, d_head//2, 2) → (B, T, H, d_head)
    x_out = torch.view_as_real(x_rotated).reshape(B, T, H, D)

    return x_out.to(x.dtype)


# ─────────────────────────────────────────────────────────────
# Causal Self-Attention
# ─────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Causal (masked) multi-head attention with RoPE.

    Attention (Vaswani et al., 2017) computes:
        Attention(Q, K, V) = softmax(QK^T / √d_head) × V

    "Causal" means we add a mask that prevents each position from
    attending to future positions. This is essential for autoregressive
    training (predicting the next token).

    "Multi-head" means we repeat the process n_heads times in different
    subspaces, then concatenate:
        MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_O

    Each head learns to attend to different types of relationships:
    some heads learn syntax, others semantics, etc.

    Detailed implementation:
        1. Project x into Q, K, V via linear transformations
        2. Apply RoPE to Q and K (not V)
        3. Compute attention with causal mask
        4. Project output back to d_model

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config   = config
        self.n_heads  = config.n_heads
        self.d_head   = config.d_head
        self.d_model  = config.d_model

        # Linear projections for Q, K, V — combined into a single matrix
        # for efficiency. Shape: (d_model) → (3 × d_model)
        # Then split into three equal parts.
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Output projection: head concatenation → d_model
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Attention dropout (regularization)
        self.attn_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular matrix of 1s
        # Registered as buffer (not a parameter, but saved in state_dict)
        # Shape: (1, 1, seq_len, seq_len) for broadcast with (B, H, T, T)
        mask = torch.tril(torch.ones(config.seq_len, config.seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, config.seq_len, config.seq_len))

    def forward(
        self,
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal multi-head attention.

        Args:
            x:             Input tensor, shape (B, T, d_model).
            freqs_complex: Pre-computed RoPE frequencies, shape (T, d_head//2).

        Returns:
            Output tensor, shape (B, T, d_model).
        """
        B, T, C = x.shape  # C = d_model

        # ── Step 1: Project into Q, K, V ─────────────────────────────────
        # qkv shape: (B, T, 3 × d_model)
        qkv = self.qkv_proj(x)

        # Split into Q, K, V: each has shape (B, T, d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, T, n_heads, d_head) to apply RoPE per head
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)

        # ── Step 2: Apply RoPE to Q and K ────────────────────────────────
        # V does not receive RoPE — position is encoded in attention via Q·K
        q = apply_rope(q, freqs_complex)
        k = apply_rope(k, freqs_complex)

        # Transpose to (B, n_heads, T, d_head) — format expected by attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── Step 3: Compute attention ─────────────────────────────────────
        if self.config.use_flash and hasattr(F, "scaled_dot_product_attention"):
            # Flash Attention (PyTorch 2.0+): more memory and speed efficient
            # Implements the same math, but without materializing
            # the full attention matrix (B, H, T, T) in memory
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,  # aplica máscara causal automaticamente
            )
        else:
            # Manual attention — more readable, useful for understanding the mechanism
            # scores shape: (B, n_heads, T, T)
            scale  = 1.0 / math.sqrt(self.d_head)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply causal mask: future positions receive -inf
            # After softmax, -inf → 0 (no attention to future tokens)
            mask = self.causal_mask[:, :, :T, :T]
            scores = scores.masked_fill(mask == 0, float("-inf"))

            # Softmax normalizes scores into a probability distribution
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            # Weighted average of values
            y = torch.matmul(attn_weights, v)

        # ── Step 4: Regroup heads and project output ─────────────────────
        # (B, n_heads, T, d_head) → (B, T, n_heads, d_head) → (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.out_proj(y)


# ─────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation (Shazeer, 2020).

    The original Transformer FFN uses two linear layers with ReLU:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    SwiGLU (Swish-Gated Linear Unit) uses a learnable "gate":
        SwiGLU(x) = (xW_1 ⊙ Swish(xW_gate)) × W_2

    Where ⊙ is element-wise multiplication and Swish(x) = x × σ(x).

    The W_gate learns to filter which activations are relevant,
    giving the model more expressive capacity at similar cost.

    Why 3 matrices instead of 2?
        SwiGLU uses 3 projections (W_1, W_gate, W_2) instead of 2.
        To maintain the same parameter count as the original FFN
        (which uses d_ff = 4 × d_model), we use d_ff ≈ 8/3 × d_model.
        In practice, we round to multiples of 256 for efficiency.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Main projection and gate projection — done together for efficiency
        # Shape: d_model → 2 × d_ff (then split in half)
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.d_ff, bias=False)

        # Output projection: d_ff → d_model
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the FFN with SwiGLU.

        Args:
            x: Tensor of shape (B, T, d_model).

        Returns:
            Tensor of shape (B, T, d_model).
        """
        # gate: passed through Swish (SiLU in PyTorch) — learns the "filter"
        # up:   main projection — the "content"
        # Element-wise multiplication is the "gating"
        gate = F.silu(self.gate_proj(x))   # Swish/SiLU: x * sigmoid(x)
        up   = self.up_proj(x)

        # Combine gate and up, project back
        hidden = self.dropout(gate * up)
        return self.down_proj(hidden)


# ─────────────────────────────────────────────────────────────
# Bloco Transformer
# ─────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Full Transformer block with pre-norm.

    Each block consists of two sub-modules with residual connections:
        1. Self-Attention (with RoPE and causal mask)
        2. Feed-Forward Network (SwiGLU)

    Pre-norm vs Post-norm:
        The original Transformer (Vaswani et al., 2017) uses post-norm:
            x = LayerNorm(x + SubLayer(x))

        Modern LLMs use pre-norm (also called "pre-LN"):
            x = x + SubLayer(LayerNorm(x))

        Pre-norm has more stable gradients during training, since
        normalization happens before non-linear transformations.
        This allows training deeper networks without extensive warm-up.

    Residual connections (He et al., 2016):
        The addition x + SubLayer(x) creates a "shortcut" that allows
        gradients to flow directly through layers, independent of
        transformations. Fundamental for training deep networks.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Normalization before attention (pre-norm)
        self.norm1 = RMSNorm(config.d_model)

        # Causal multi-head attention with RoPE
        self.attn  = CausalSelfAttention(config)

        # Normalization before FFN (pre-norm)
        self.norm2 = RMSNorm(config.d_model)

        # Feed-forward with SwiGLU
        self.ffn   = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process x through the Transformer block.

        Args:
            x:             Tensor of shape (B, T, d_model).
            freqs_complex: RoPE frequencies of shape (T, d_head//2).

        Returns:
            Tensor of shape (B, T, d_model).
        """
        # Sub-block 1: attention with residual connection
        # Pre-norm: normalize x before passing through attention
        x = x + self.attn(self.norm1(x), freqs_complex)

        # Sub-block 2: FFN with residual connection
        x = x + self.ffn(self.norm2(x))

        return x


# ─────────────────────────────────────────────────────────────
# Modelo completo
# ─────────────────────────────────────────────────────────────

class MiniLM(nn.Module):
    """
    Complete Transformer Decoder-only language model.

    "MiniLM" is the name given to this project's model. Architecture
    based on modern best practices (LLaMA-style).

    Components (in forward pass order):
        1. Token Embedding: maps token IDs to dense vectors
        2. N × TransformerBlock: processes vectors with attention and FFN
        3. Final RMSNorm: normalizes before output projection
        4. LM Head: projects from d_model to vocab_size (logits)

    Weight tying:
        Input embedding and LM head weights are shared (tied weights).
        This reduces parameter count by ~10-20% without quality loss —
        used in GPT-2 and LLaMA.
        Intuition: the embedding learns "what tokens look like", and
        the LM head learns "which tokens are likely" — similar information.

    Args:
        config: Full model configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Token embedding ───────────────────────────────────────────────
        # Maps integer IDs (0..vocab_size-1) to d_model-dimensional vectors
        # Weight shape: (vocab_size, d_model)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # ── Embedding dropout ─────────────────────────────────────────────
        self.emb_dropout = nn.Dropout(config.dropout)

        # ── Transformer block stack ───────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        # ── Final normalization ───────────────────────────────────────────
        self.norm_final = RMSNorm(config.d_model)

        # ── LM Head ───────────────────────────────────────────────────────
        # Projects d_model → vocab_size to obtain logits (no bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between embedding and lm_head
        # Both have shape (vocab_size, d_model) — they are the same tensor
        self.lm_head.weight = self.token_emb.weight

        # ── RoPE pre-computation ─────────────────────────────────────────
        # Compute rotation frequencies once, for all positions
        # Registered as buffer: saved in checkpoint, but not a parameter
        freqs = precompute_rope_freqs(config.d_head, config.seq_len)
        self.register_buffer("freqs_complex", freqs)

        # ── Weight initialization ─────────────────────────────────────────
        self.apply(self._init_weights)

        # Special initialization for residual projections (GPT-2 style):
        # scale by number of layers to stabilize gradients
        for name, param in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layers)
                )

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize model weights.

        Follows GPT-2 initialization:
            - Linear and Embedding layers: Normal(0, 0.02)
            - Bias (when present): zeros

        The Normal(0, 0.02) distribution is small enough to keep
        activations at a reasonable scale at the start of training,
        avoiding gradient explosion or vanishing.

        Args:
            module: Module to initialize (called recursively by apply()).
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Model forward pass.

        Training mode (targets provided):
            Computes logits AND loss efficiently in a single forward pass.

        Inference mode (targets=None):
            Returns only the last position logits.

        Args:
            input_ids: Token ID tensor, shape (B, T).
            targets:   Next token IDs, shape (B, T).
                       If provided, computes the cross-entropy loss.

        Returns:
            Tuple (logits, loss):
                logits: shape (B, T, vocab_size) — raw probabilities
                loss:   scalar if targets provided, None otherwise

        Training example:
            input_ids = [t1, t2, t3, t4]   ← input tokens
            targets   = [t2, t3, t4, t5]   ← next tokens (shift of 1)
            The model learns: given t1, predict t2; given t1,t2, predict t3; etc.
        """
        B, T = input_ids.shape
        assert T <= self.config.seq_len, (
            f"Sequence of length {T} exceeds seq_len={self.config.seq_len}"
        )

        # ── Token embedding ───────────────────────────────────────────────
        # (B, T) → (B, T, d_model)
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)

        # ── RoPE frequencies for the current T positions ─────────────────
        # Slicing: take only the first T positions (important for
        # incremental generation where T < seq_len)
        freqs = self.freqs_complex[:T]

        # ── Pass through Transformer blocks ──────────────────────────────
        for block in self.blocks:
            x = block(x, freqs)

        # ── Final normalization ───────────────────────────────────────────
        x = self.norm_final(x)

        # ── LM Head ───────────────────────────────────────────────────────
        if targets is not None:
            # Training mode: compute logits for all positions
            # (B, T, d_model) → (B, T, vocab_size)
            logits = self.lm_head(x)

            # Cross-entropy loss: flatten (B, T, vocab_size) → (B*T, vocab_size)
            # and targets (B, T) → (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # -1 is used to mask padding positions
            )
            return logits, loss
        else:
            # Inference mode: compute logits only for the last token
            # More efficient — intermediate logits are not needed
            logits = self.lm_head(x[:, -1:, :])
            return logits, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        The generation process works in a loop:
            1. Pass the current sequence through the model → next token logits
            2. Apply temperature (controls randomness)
            3. Apply top-k and/or top-p filters (controls diversity)
            4. Sample the next token
            5. Append to sequence and repeat

        Temperature:
            - T → 0: deterministic generation (always the most probable token)
            - T = 1: original model distribution
            - T > 1: more random, more creative (but may be incoherent)

        Top-k sampling:
            Keeps only the k most probable tokens before sampling.
            Prevents very unlikely tokens from being selected.

        Top-p (nucleus) sampling (Holtzman et al., 2019):
            Keeps the smallest set of tokens whose cumulative probability
            ≥ p. Adaptively selects more or fewer tokens depending on
            the distribution.

        Args:
            input_ids:      Initial context tokens, shape (1, T).
            max_new_tokens: How many new tokens to generate.
            temperature:    Randomness control (0.1 to 2.0).
            top_k:          Filter to top-k tokens (e.g., 50).
            top_p:          Nucleus sampling (e.g., 0.9).

        Returns:
            Tensor with full sequence (context + generated), shape (1, T+N).
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Truncate context if it exceeds seq_len
            context = input_ids[:, -self.config.seq_len:]

            # Forward pass — only the last token logits
            logits, _ = self(context)
            # logits shape: (1, 1, vocab_size) → (vocab_size,)
            logits = logits[:, -1, :].squeeze(0)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k: zero out logits outside top-k
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                threshold = values[-1]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Apply top-p (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens above the cumulative threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token (shift right)
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float("-inf")

            # Convert logits to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

            # Append the new token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> dict[str, int]:
        """
        Count model parameters by component.

        Useful for verifying the parameter distribution and understanding
        where model capacity is concentrated.

        Returns:
            Dictionary with parameter count per component.
        """
        def count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "token_embedding":  count(self.token_emb),
            "attention_layers": sum(count(b.attn) for b in self.blocks),
            "ffn_layers":       sum(count(b.ffn)  for b in self.blocks),
            "norm_layers":      sum(count(b.norm1) + count(b.norm2) for b in self.blocks),
            "lm_head":          0,  # tied weights — not counted twice
            "total":            count(self),
        }

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"MiniLM(\n"
            f"  vocab_size={self.config.vocab_size}, "
            f"seq_len={self.config.seq_len}\n"
            f"  d_model={self.config.d_model}, "
            f"n_heads={self.config.n_heads}, "
            f"n_layers={self.config.n_layers}\n"
            f"  d_ff={self.config.d_ff}, "
            f"d_head={self.config.d_head}\n"
            f"  params={params['total'] / 1e6:.1f}M\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────
# Utilitários de VRAM
# ─────────────────────────────────────────────────────────────

def estimate_vram(config: ModelConfig, batch_size: int = 8, dtype_bytes: int = 2) -> dict:
    """
    Estimate VRAM usage for training the model.

    Total training memory has four components:
        1. Model parameters
        2. Gradients (same size as parameters)
        3. Optimizer states (AdamW keeps 2 moments per parameter)
        4. Activations (depends on batch size and seq_len)

    This is a conservative estimate — actual usage may vary.

    Args:
        config:      Model configuration.
        batch_size:  Training batch size.
        dtype_bytes: Bytes per parameter (2 for bf16/fp16, 4 for fp32).

    Returns:
        Dictionary with GB estimates per component.
    """
    n_params = config.n_params

    # Parameters + gradients (same dtype)
    params_gb   = n_params * dtype_bytes / 1e9
    grads_gb    = params_gb

    # AdamW: 2 moments in fp32 (8 bytes per parameter)
    optimizer_gb = n_params * 8 / 1e9

    # Activations (approximate estimate)
    # Each block stores: x, attn_weights, ffn_hidden
    activations_per_block = batch_size * config.seq_len * config.d_model * dtype_bytes
    activations_gb = config.n_layers * activations_per_block / 1e9

    total_gb = params_gb + grads_gb + optimizer_gb + activations_gb

    return {
        "parameters":    f"{params_gb:.2f} GB",
        "gradients":     f"{grads_gb:.2f} GB",
        "optimizer":     f"{optimizer_gb:.2f} GB",
        "activations":   f"{activations_gb:.2f} GB",
        "total_estimate":f"{total_gb:.2f} GB",
        "n_params":      f"{n_params / 1e6:.1f}M",
    }


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MiniLM Demo")
    print("=" * 60)

    # Small configuration (~85M parameters)
    config = ModelConfig(
        vocab_size=16384,
        seq_len=512,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=1536,
        dropout=0.1,
    )

    print(f"\nAvailable configurations:")
    configs = {
        "Tiny  (~15M)": ModelConfig(d_model=256, n_heads=4,  n_layers=4,  d_ff=768),
        "Small (~85M)": ModelConfig(d_model=512, n_heads=8,  n_layers=8,  d_ff=1536),
        "Base (~310M)": ModelConfig(d_model=768, n_heads=12, n_layers=12, d_ff=2304),
    }
    for name, cfg in configs.items():
        print(f"  {name}: {cfg.n_params / 1e6:.0f}M params")

    print(f"\nInstantiating Small model...")
    model = MiniLM(config)
    print(model)

    # Contagem detalhada de parâmetros
    print("\nParameter distribution:")
    for component, count in model.count_parameters().items():
        if count > 0:
            print(f"  {component:<20}: {count / 1e6:.2f}M")

    # Estimativa de VRAM
    print("\nVRAM estimate (batch=8, bf16):")
    vram = estimate_vram(config, batch_size=8, dtype_bytes=2)
    for k, v in vram.items():
        print(f"  {k:<20}: {v}")

    # Teste de forward pass
    print("\nForward pass test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    model = model.to(device)

    B, T = 2, 64  # batch_size=2, seq_len=64
    input_ids = torch.randint(0, config.vocab_size, (B, T)).to(device)
    targets   = torch.randint(0, config.vocab_size, (B, T)).to(device)

    logits, loss = model(input_ids, targets)
    print(f"  Input shape  : {input_ids.shape}")
    print(f"  Logits shape : {logits.shape}")
    print(f"  Initial loss : {loss.item():.4f}")
    print(f"  Expected loss: {math.log(config.vocab_size):.4f} (maximum entropy)")

    # Teste de geração
    print("\nGeneration test (10 tokens)...")
    prompt = torch.randint(0, config.vocab_size, (1, 5)).to(device)
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f"  Prompt shape   : {prompt.shape}")
    print(f"  Generated shape: {generated.shape}")
    print(f"  New tokens     : {generated[0, 5:].tolist()}")
    print("\nForward pass and generation OK.")
