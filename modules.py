"""
Core modules for HRM: Transformer blocks, L-module, H-module, embeddings, and output heads.
Follows Llama-style architecture with RoPE, GLU, RMSNorm, and Post-Norm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no scale/bias parameters)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Note: Paper specifies scale and bias are excluded from RMSNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for cos/sin values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        self._update_cache(seq_len, q.device)

        cos = self._cos_cached[:seq_len, :]
        sin = self._sin_cached[:seq_len, :]

        return self._apply_rotary_emb(q, cos, sin), self._apply_rotary_emb(k, cos, sin)

    @staticmethod
    def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_heads, head_dim]
        # cos, sin: [seq_len, head_dim]
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and no bias terms."""

    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # No bias terms (following paper)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]

        # Apply rotary embeddings
        q, k = self.rope(q.transpose(1, 2), k.transpose(1, 2))  # Need [batch, seq_len, n_heads, head_dim]
        q = q.transpose(1, 2)  # Back to [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.dim)
        return self.out_proj(out)


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) feedforward network."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        # No bias terms
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # SiLU activation
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """
    Transformer block with Post-Norm architecture.
    Follows Llama-style: RoPE, GLU, RMSNorm, no bias terms.
    """

    def __init__(self, dim: int, n_heads: int, hidden_dim: Optional[int] = None, max_seq_len: int = 2048):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, max_seq_len)
        self.ffn = GatedLinearUnit(dim, hidden_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Post-Norm: x -> sublayer -> norm -> add
        x = self.norm1(x + self.attention(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


class InputEmbedding(nn.Module):
    """Input embedding for RNA sequences."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Truncated LeCun Normal initialization
        nn.init.trunc_normal_(self.embed.weight, std=1.0 / math.sqrt(embed_dim), a=-2.0, b=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class OutputHead(nn.Module):
    """Output head for structure prediction."""

    def __init__(self, hidden_dim: int, output_vocab_size: int, use_stablemax: bool = False):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, output_vocab_size, bias=False)
        self.use_stablemax = use_stablemax
        # Truncated LeCun Normal initialization
        nn.init.trunc_normal_(self.proj.weight, std=1.0 / math.sqrt(hidden_dim), a=-2.0, b=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.proj(x)

        if self.use_stablemax:
            # Stablemax: more stable than softmax for small sample experiments
            # stablemax(x) = softmax(x - max(x) + log(n))
            n = logits.shape[-1]
            logits = logits - logits.max(dim=-1, keepdim=True).values + math.log(n)

        return F.softmax(logits, dim=-1)


class LModule(nn.Module):
    """
    Low-level recurrent module.
    Updates at each timestep conditioned on H-module state and input.
    """

    def __init__(self, dim: int, n_heads: int, n_layers: int = 1, max_seq_len: int = 2048):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])

    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_L: Previous L-module state [batch, seq_len, dim]
            z_H: Current H-module state [batch, seq_len, dim]
            x_tilde: Input representation [batch, seq_len, dim]

        Returns:
            Updated L-module state [batch, seq_len, dim]
        """
        # Combine inputs via element-wise addition (paper mentions this is simple approach)
        z = z_L + z_H + x_tilde

        for layer in self.layers:
            z = layer(z)

        return z


class HModule(nn.Module):
    """
    High-level recurrent module.
    Updates once per cycle using L-module's final state.
    """

    def __init__(self, dim: int, n_heads: int, n_layers: int = 1, max_seq_len: int = 2048):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])

    def forward(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_H: Previous H-module state [batch, seq_len, dim]
            z_L: Final L-module state from current cycle [batch, seq_len, dim]

        Returns:
            Updated H-module state [batch, seq_len, dim]
        """
        # Combine inputs via element-wise addition
        z = z_H + z_L

        for layer in self.layers:
            z = layer(z)

        return z


class QHead(nn.Module):
    """Q-learning head for Adaptive Computational Time (ACT)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Predicts Q-values for halt and continue actions
        self.proj = nn.Linear(hidden_dim, 2, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=1.0 / math.sqrt(hidden_dim), a=-2.0, b=2.0)

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_H: H-module state [batch, seq_len, dim]

        Returns:
            Q-values [batch, 2] for [halt, continue]
        """
        # Pool over sequence dimension (mean pooling)
        pooled = z_H.mean(dim=1)  # [batch, dim]
        q_values = torch.sigmoid(self.proj(pooled))  # [batch, 2]
        return q_values
