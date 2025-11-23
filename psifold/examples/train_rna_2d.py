"""
Example training script for HRM on RNA 2D structure prediction.

This demonstrates:
1. Setting up the HRM model
2. Generating synthetic training data
3. Training with deep supervision and ACT
4. Evaluating the model
5. Analyzing hierarchical convergence
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from psifold import HRM, HRMTrainer
from psifold.model import HRMConfig
from psifold.training import create_optimizer, LRScheduler
from psifold.utils import (
    generate_synthetic_data,
    collate_batch,
    calculate_structure_accuracy,
    visualize_prediction,
    decode_rna_sequence,
    decode_structure,
    RNA_VOCAB,
    STRUCTURE_VOCAB,
)


class RNADataset(Dataset):
    """Simple dataset for RNA sequence-structure pairs."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch(trainer, dataloader, scheduler, epoch):
    """Train for one epoch."""
    trainer.model.train()
    total_loss = 0.0
    total_seq_loss = 0.0
    total_q_loss = 0.0
    total_segments = 0.0
    n_batches = 0

    for batch_idx, (sequences, structures) in enumerate(dataloader):
        # Train step
        metrics = trainer.train_step(sequences, structures, ignore_index=STRUCTURE_VOCAB['<PAD>'])

        # Update learning rate
        scheduler.step()

        # Accumulate metrics
        total_loss += metrics['loss']
        total_seq_loss += metrics['seq_loss']
        total_q_loss += metrics['q_loss']
        total_segments += metrics['avg_segments']
        n_batches += 1

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"Loss={metrics['loss']:.4f}, "
                  f"SeqLoss={metrics['seq_loss']:.4f}, "
                  f"QLoss={metrics['q_loss']:.4f}, "
                  f"AvgSegments={metrics['avg_segments']:.2f}, "
                  f"LR={scheduler.get_last_lr()[0]:.6f}")

    # Return average metrics
    return {
        'loss': total_loss / n_batches,
        'seq_loss': total_seq_loss / n_batches,
        'q_loss': total_q_loss / n_batches,
        'avg_segments': total_segments / n_batches,
    }


def eval_epoch(trainer, dataloader):
    """Evaluate for one epoch."""
    trainer.model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_segments = 0.0
    n_batches = 0

    for sequences, structures in dataloader:
        # Eval step
        metrics = trainer.eval_step(sequences, structures, ignore_index=STRUCTURE_VOCAB['<PAD>'])

        # Accumulate metrics
        total_loss += metrics['loss']
        total_accuracy += metrics['accuracy']
        total_segments += metrics['segments_used']
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'accuracy': total_accuracy / n_batches,
        'avg_segments': total_segments / n_batches,
    }


def visualize_hierarchical_convergence(model, sample_data):
    """Visualize hierarchical convergence on a sample."""
    model.eval()

    # Get a sample
    sequences, structures = collate_batch([sample_data], max_len=50)
    sequences = sequences.to(next(model.parameters()).device)

    # Run verbose forward pass
    with torch.no_grad():
        results = model.forward_verbose(sequences)

    # Print convergence info
    print("\n=== Hierarchical Convergence Analysis ===")
    print(f"L-module residuals (per timestep):")
    for i, residual in enumerate(results['l_residuals']):
        print(f"  Step {i + 1}: {residual:.6f}")

    print(f"\nH-module residuals (per cycle):")
    for i, residual in enumerate(results['h_residuals']):
        print(f"  Cycle {i + 1}: {residual:.6f}")

    # Visualize prediction
    seq_str = decode_rna_sequence(sequences[0].cpu().tolist())
    true_str = sample_data[1]
    pred_tokens = results['y_hat'][0].argmax(dim=-1).cpu().tolist()
    pred_str = decode_structure(pred_tokens)

    print(f"\n=== Prediction ===")
    print(visualize_prediction(seq_str, true_str, pred_str))


def main():
    """Main training loop."""
    print("=" * 60)
    print("HRM for RNA 2D Structure Prediction")
    print("=" * 60)

    # Configuration
    config = HRMConfig(
        vocab_size=len(RNA_VOCAB),
        output_vocab_size=len(STRUCTURE_VOCAB),
        dim=128,  # Smaller for faster training
        n_heads=4,
        n_cycles=2,
        cycle_steps=3,
        l_layers=2,
        h_layers=2,
        max_seq_len=512,
        use_stablemax=True,  # Better for small samples
        use_act=True,
    )

    # Create model
    print("\nCreating HRM model...")
    model = config.create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    train_data = generate_synthetic_data(n_samples=500, min_len=10, max_len=50, seed=42)
    val_data = generate_synthetic_data(n_samples=100, min_len=10, max_len=50, seed=43)

    # Create datasets and dataloaders
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, max_len=50),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, max_len=50),
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, lr=1e-3, weight_decay=0.01, warmup_steps=100)
    scheduler = LRScheduler(optimizer, warmup_steps=100, base_lr=1e-3)

    # Create trainer
    trainer = HRMTrainer(
        model=model,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_segments=8,
        epsilon=0.1,
        use_act=True,
    )

    print(f"\nTraining on device: {trainer.device}")

    # Training loop
    n_epochs = 10

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, n_epochs + 1):
        print(f"\n--- Epoch {epoch}/{n_epochs} ---")

        # Train
        train_metrics = train_epoch(trainer, train_loader, scheduler, epoch)
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"SeqLoss: {train_metrics['seq_loss']:.4f}, "
              f"AvgSegments: {train_metrics['avg_segments']:.2f}")

        # Evaluate
        val_metrics = eval_epoch(trainer, val_loader)
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}, "
              f"AvgSegments: {val_metrics['avg_segments']:.2f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Analyze hierarchical convergence
    print("\nAnalyzing hierarchical convergence on sample...")
    visualize_hierarchical_convergence(model, val_data[0])

    # Test inference-time scaling
    print("\n" + "=" * 60)
    print("Testing inference-time scaling...")
    print("=" * 60)

    for max_segs in [4, 8, 12, 16]:
        val_metrics = eval_epoch(trainer, val_loader)
        trainer.max_segments = max_segs
        print(f"Max segments = {max_segs}: Accuracy = {val_metrics['accuracy']:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
