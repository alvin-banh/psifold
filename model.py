"""
Hierarchical Recurrent Model (HRM) for RNA 2D Structure Prediction.

Implements the core HRM architecture with:
- Hierarchical processing (L-module and H-module)
- 1-step gradient approximation (no BPTT)
- Support for deep supervision and ACT
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .modules import InputEmbedding, LModule, HModule, OutputHead, QHead


class HRM(nn.Module):
    """
    Hierarchical Recurrent Model.

    The model processes inputs through N high-level cycles of T low-level timesteps each.
    This creates hierarchical convergence where the L-module converges within each cycle
    while the H-module guides the overall computation.

    Args:
        vocab_size: Size of input vocabulary (e.g., 4 for RNA: A, U, G, C)
        output_vocab_size: Size of output vocabulary (e.g., for structure tokens)
        dim: Hidden dimension for both L and H modules
        n_heads: Number of attention heads
        n_cycles: Number of high-level cycles (N)
        cycle_steps: Number of low-level steps per cycle (T)
        l_layers: Number of Transformer layers in L-module
        h_layers: Number of Transformer layers in H-module
        max_seq_len: Maximum sequence length
        use_stablemax: Whether to use stablemax instead of softmax (better for small samples)
        use_act: Whether to use Adaptive Computational Time
    """

    def __init__(
        self,
        vocab_size: int,
        output_vocab_size: int,
        dim: int = 256,
        n_heads: int = 8,
        n_cycles: int = 2,
        cycle_steps: int = 2,
        l_layers: int = 2,
        h_layers: int = 2,
        max_seq_len: int = 512,
        use_stablemax: bool = False,
        use_act: bool = True,
    ):
        super().__init__()

        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size
        self.dim = dim
        self.n_cycles = n_cycles
        self.cycle_steps = cycle_steps
        self.max_seq_len = max_seq_len
        self.use_act = use_act

        # Four learnable components
        self.input_embedding = InputEmbedding(vocab_size, dim)
        self.l_module = LModule(dim, n_heads, l_layers, max_seq_len)
        self.h_module = HModule(dim, n_heads, h_layers, max_seq_len)
        self.output_head = OutputHead(dim, output_vocab_size, use_stablemax)

        # Q-head for ACT
        if use_act:
            self.q_head = QHead(dim)

        # Initialize hidden states
        self._init_hidden_states()

    def _init_hidden_states(self):
        """Initialize z_0 using truncated normal distribution (std=1, truncation=2)."""
        # These are registered as buffers and kept fixed during training
        z_init = torch.randn(1, 1, self.dim)
        z_init = torch.clamp(z_init, -2.0, 2.0)  # Truncation

        self.register_buffer("z_L_init", z_init)
        self.register_buffer("z_H_init", z_init)

    def get_initial_state(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get initial hidden states for a batch.

        Returns:
            Tuple of (z_L_0, z_H_0) with shape [batch, seq_len, dim]
        """
        z_L = self.z_L_init.expand(batch_size, seq_len, self.dim).clone()
        z_H = self.z_H_init.expand(batch_size, seq_len, self.dim).clone()
        return z_L, z_H

    def forward(
        self,
        x: torch.Tensor,
        z_L: Optional[torch.Tensor] = None,
        z_H: Optional[torch.Tensor] = None,
        return_q_values: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Single forward pass of HRM (N cycles of T steps = N*T total timesteps).

        This implements the 1-step gradient approximation:
        - All timesteps except the last are computed with torch.no_grad()
        - Only the final L and H updates have gradients enabled
        - This gives O(1) memory instead of O(N*T)

        Args:
            x: Input sequence [batch, seq_len] of token indices
            z_L: Initial L-module state [batch, seq_len, dim] (optional)
            z_H: Initial H-module state [batch, seq_len, dim] (optional)
            return_q_values: Whether to return Q-values for ACT

        Returns:
            Tuple of:
            - y_hat: Predicted output [batch, seq_len, output_vocab_size]
            - (z_L_final, z_H_final): Final hidden states
            - q_values: Q-values [batch, 2] for [halt, continue] (if return_q_values=True)
        """
        batch_size, seq_len = x.shape

        # Initialize states if not provided
        if z_L is None or z_H is None:
            z_L, z_H = self.get_initial_state(batch_size, seq_len)

        # Input embedding (this has gradients)
        x_tilde = self.input_embedding(x)

        total_steps = self.n_cycles * self.cycle_steps

        # NO GRADIENTS for all but the last step (1-step approximation)
        with torch.no_grad():
            for i in range(total_steps - 1):
                # L-module updates every timestep
                z_L = self.l_module(z_L, z_H, x_tilde)

                # H-module updates every T timesteps (at end of each cycle)
                if (i + 1) % self.cycle_steps == 0:
                    z_H = self.h_module(z_H, z_L)

        # FINAL STEP with gradients (this is the 1-step gradient approximation)
        z_L = self.l_module(z_L, z_H, x_tilde)
        z_H = self.h_module(z_H, z_L)

        # Output prediction from final H-module state
        y_hat = self.output_head(z_H)

        # Q-values for ACT
        q_values = None
        if return_q_values and self.use_act:
            q_values = self.q_head(z_H)

        return y_hat, (z_L, z_H), q_values

    def forward_verbose(
        self,
        x: torch.Tensor,
        z_L: Optional[torch.Tensor] = None,
        z_H: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass with detailed information about hierarchical convergence.
        Useful for analysis and visualization.

        Returns dictionary with:
        - y_hat: Output prediction
        - states: List of (z_L, z_H) at each timestep
        - l_residuals: L-module residuals at each step
        - h_residuals: H-module residuals at each cycle
        """
        batch_size, seq_len = x.shape

        if z_L is None or z_H is None:
            z_L, z_H = self.get_initial_state(batch_size, seq_len)

        x_tilde = self.input_embedding(x)

        states = [(z_L.clone(), z_H.clone())]
        l_residuals = []
        h_residuals = []

        total_steps = self.n_cycles * self.cycle_steps

        for i in range(total_steps):
            z_L_prev = z_L.clone()
            z_L = self.l_module(z_L, z_H, x_tilde)
            l_residual = (z_L - z_L_prev).pow(2).mean().item()
            l_residuals.append(l_residual)

            if (i + 1) % self.cycle_steps == 0:
                z_H_prev = z_H.clone()
                z_H = self.h_module(z_H, z_L)
                h_residual = (z_H - z_H_prev).pow(2).mean().item()
                h_residuals.append(h_residual)

            states.append((z_L.clone(), z_H.clone()))

        y_hat = self.output_head(z_H)

        return {
            "y_hat": y_hat,
            "states": states,
            "l_residuals": l_residuals,
            "h_residuals": h_residuals,
        }


class HRMConfig:
    """Configuration for HRM model."""

    def __init__(
        self,
        vocab_size: int = 5,  # A, U, G, C, + padding
        output_vocab_size: int = 4,  # Structure tokens: . ( ) and padding
        dim: int = 256,
        n_heads: int = 8,
        n_cycles: int = 2,
        cycle_steps: int = 2,
        l_layers: int = 2,
        h_layers: int = 2,
        max_seq_len: int = 512,
        use_stablemax: bool = False,
        use_act: bool = True,
    ):
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_cycles = n_cycles
        self.cycle_steps = cycle_steps
        self.l_layers = l_layers
        self.h_layers = h_layers
        self.max_seq_len = max_seq_len
        self.use_stablemax = use_stablemax
        self.use_act = use_act

    def create_model(self) -> HRM:
        """Create HRM model from config."""
        return HRM(
            vocab_size=self.vocab_size,
            output_vocab_size=self.output_vocab_size,
            dim=self.dim,
            n_heads=self.n_heads,
            n_cycles=self.n_cycles,
            cycle_steps=self.cycle_steps,
            l_layers=self.l_layers,
            h_layers=self.h_layers,
            max_seq_len=self.max_seq_len,
            use_stablemax=self.use_stablemax,
            use_act=self.use_act,
        )
