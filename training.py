"""
Training utilities for HRM with Deep Supervision and Adaptive Computational Time (ACT).

Implements:
- Deep supervision: Multiple forward passes with gradient detachment between segments
- ACT with Q-learning: Adaptive halting mechanism
- Training loop and utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import random

from .model import HRM


class HRMTrainer:
    """
    Trainer for HRM with deep supervision and adaptive computational time.

    The training process works as follows:
    1. For each sample, run multiple forward passes (segments)
    2. After each segment, compute loss and update parameters (deep supervision)
    3. Detach hidden state before next segment (1-step gradient approximation)
    4. Use Q-learning to learn when to halt (ACT)
    """

    def __init__(
        self,
        model: HRM,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_segments: int = 8,
        epsilon: float = 0.1,
        use_act: bool = True,
    ):
        """
        Args:
            model: HRM model
            optimizer: Optimizer (e.g., AdamW with constant lr + warmup)
            device: Device to run on
            max_segments: Maximum number of segments (M_max)
            epsilon: Exploration probability for ACT
            use_act: Whether to use adaptive computational time
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.max_segments = max_segments
        self.epsilon = epsilon
        self.use_act = use_act

    def compute_sequence_loss(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Compute sequence-to-sequence loss (averaged over tokens).

        Args:
            y_hat: Predicted probabilities [batch, seq_len, vocab_size]
            y_true: True labels [batch, seq_len]
            ignore_index: Index to ignore in loss computation

        Returns:
            Loss scalar
        """
        batch_size, seq_len, vocab_size = y_hat.shape

        # Reshape for cross entropy
        y_hat_flat = y_hat.reshape(-1, vocab_size)
        y_true_flat = y_true.reshape(-1)

        # Cross entropy with log probabilities
        loss = F.cross_entropy(
            torch.log(y_hat_flat + 1e-10),  # Add epsilon for numerical stability
            y_true_flat,
            ignore_index=ignore_index,
        )

        return loss

    def compute_q_loss(
        self,
        q_values: torch.Tensor,
        q_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-learning loss.

        Args:
            q_values: Predicted Q-values [batch, 2]
            q_targets: Target Q-values [batch, 2]

        Returns:
            Loss scalar
        """
        return F.binary_cross_entropy(q_values, q_targets)

    def should_halt(
        self,
        q_values: torch.Tensor,
        segment: int,
        min_segments: int,
    ) -> torch.Tensor:
        """
        Determine whether to halt for each sample in batch.

        Args:
            q_values: Q-values [batch, 2] for [halt, continue]
            segment: Current segment number
            min_segments: Minimum number of segments for this batch

        Returns:
            Boolean tensor [batch] indicating whether to halt
        """
        # Halt if Q(halt) > Q(continue) and segment >= min_segments
        q_halt, q_continue = q_values[:, 0], q_values[:, 1]
        should_halt = (q_halt > q_continue) & (segment >= min_segments)

        # Always halt at max_segments
        if segment >= self.max_segments:
            should_halt = torch.ones_like(should_halt, dtype=torch.bool)

        return should_halt

    def sample_min_segments(self) -> int:
        """
        Sample minimum number of segments.

        With probability epsilon, sample uniformly from {2, ..., M_max}
        With probability 1-epsilon, return 1
        """
        if random.random() < self.epsilon:
            return random.randint(2, self.max_segments)
        else:
            return 1

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ignore_index: int = -100,
    ) -> Dict[str, Any]:
        """
        Single training step with deep supervision and ACT.

        Args:
            x: Input sequences [batch, seq_len]
            y: Target sequences [batch, seq_len]
            ignore_index: Index to ignore in loss computation

        Returns:
            Dictionary with training metrics
        """
        x = x.to(self.device)
        y = y.to(self.device)

        batch_size = x.shape[0]

        # Initialize tracking
        total_loss = 0.0
        total_seq_loss = 0.0
        total_q_loss = 0.0
        segments_used = []

        # Get initial states
        z_L, z_H = self.model.get_initial_state(batch_size, x.shape[1])

        # Sample minimum segments
        min_segments = self.sample_min_segments()

        # Track which samples are still active (not halted)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        # Deep supervision loop
        for segment in range(1, self.max_segments + 1):
            if not active_mask.any():
                break  # All samples halted

            # Forward pass
            y_hat, (z_L, z_H), q_values = self.model(
                x,
                z_L=z_L,
                z_H=z_H,
                return_q_values=self.use_act,
            )

            # Compute sequence loss
            seq_loss = self.compute_sequence_loss(y_hat, y, ignore_index)

            # Compute Q-targets if using ACT
            if self.use_act and q_values is not None:
                # Check if predictions are correct
                y_hat_tokens = y_hat.argmax(dim=-1)
                correct = (y_hat_tokens == y).all(dim=1).float()  # [batch]

                # Q-target for halt action: immediate reward
                q_target_halt = correct

                # Q-target for continue action: next Q-value (if not at max)
                if segment < self.max_segments:
                    # Will be filled in next iteration
                    # For now, use max of next Q-values (this is approximate)
                    q_target_continue = torch.zeros_like(correct)
                else:
                    q_target_continue = correct

                q_targets = torch.stack([q_target_halt, q_target_continue], dim=1)

                # Compute Q-loss
                q_loss = self.compute_q_loss(q_values, q_targets)

                # Total loss for this segment
                loss = seq_loss + q_loss
                total_q_loss += q_loss.item()
            else:
                loss = seq_loss
                q_values = None

            # Backward pass and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_seq_loss += seq_loss.item()

            # Detach hidden states (crucial for 1-step gradient approximation)
            z_L = z_L.detach()
            z_H = z_H.detach()

            # Determine which samples should halt
            if self.use_act and q_values is not None:
                halt_mask = self.should_halt(q_values, segment, min_segments)
                active_mask = active_mask & ~halt_mask

                # Track segments used for each sample
                for i in range(batch_size):
                    if halt_mask[i] and len(segments_used) <= i:
                        segments_used.append(segment)
            else:
                # Without ACT, always use max_segments
                if segment == self.max_segments:
                    segments_used = [self.max_segments] * batch_size

        # Compute average segments used
        avg_segments = sum(segments_used) / len(segments_used) if segments_used else self.max_segments

        return {
            "loss": total_loss / len(segments_used) if segments_used else total_loss,
            "seq_loss": total_seq_loss / len(segments_used) if segments_used else total_seq_loss,
            "q_loss": total_q_loss / len(segments_used) if segments_used and self.use_act else 0.0,
            "avg_segments": avg_segments,
            "min_segments": min_segments,
        }

    def eval_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        max_segments: Optional[int] = None,
        ignore_index: int = -100,
    ) -> Dict[str, Any]:
        """
        Single evaluation step.

        Args:
            x: Input sequences [batch, seq_len]
            y: Target sequences [batch, seq_len]
            max_segments: Maximum segments to use (defaults to self.max_segments)
            ignore_index: Index to ignore in loss computation

        Returns:
            Dictionary with evaluation metrics
        """
        x = x.to(self.device)
        y = y.to(self.device)

        if max_segments is None:
            max_segments = self.max_segments

        batch_size = x.shape[0]

        # Get initial states
        z_L, z_H = self.model.get_initial_state(batch_size, x.shape[1])

        # Track best predictions
        best_loss = float('inf')
        best_y_hat = None
        segments_used = []

        with torch.no_grad():
            for segment in range(1, max_segments + 1):
                # Forward pass
                y_hat, (z_L, z_H), q_values = self.model(
                    x,
                    z_L=z_L,
                    z_H=z_H,
                    return_q_values=self.use_act,
                )

                # Compute loss
                seq_loss = self.compute_sequence_loss(y_hat, y, ignore_index)

                # Track best
                if seq_loss < best_loss:
                    best_loss = seq_loss
                    best_y_hat = y_hat

                # Check halting
                if self.use_act and q_values is not None:
                    halt_mask = self.should_halt(q_values, segment, min_segments=1)
                    if halt_mask.all():
                        segments_used = [segment] * batch_size
                        break

        # Compute accuracy
        y_hat_tokens = best_y_hat.argmax(dim=-1)
        correct = (y_hat_tokens == y).float()
        accuracy = correct.mean().item()

        return {
            "loss": best_loss.item(),
            "accuracy": accuracy,
            "segments_used": sum(segments_used) / len(segments_used) if segments_used else max_segments,
        }


def create_optimizer(
    model: HRM,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
) -> torch.optim.Optimizer:
    """
    Create optimizer for HRM.

    The paper uses Adam-atan2 (scale-invariant variant of Adam).
    Here we use AdamW which is more standard and includes weight decay.

    Args:
        model: HRM model
        lr: Learning rate (constant after warmup)
        weight_decay: Weight decay (for L∞ bounded parameters)
        warmup_steps: Number of warmup steps

    Returns:
        Optimizer
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


class LRScheduler:
    """Learning rate scheduler with linear warmup and constant lr."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0

    def step(self):
        """Update learning rate."""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Constant lr
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
