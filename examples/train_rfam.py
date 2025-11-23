"""
Train HRM on Rfam dataset for RNA 2D structure prediction.

This script:
1. Loads Rfam.csv data
2. Splits into train/val/test sets (80/10/10)
3. Trains HRM with deep supervision and ACT
4. Evaluates on test set with F1, Precision, Recall, MCC
5. Compares against benchmarks
6. Saves best model

Usage:
    python examples/train_rfam.py --data_path /path/to/Rfam.csv
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from psifold import HRM, HRMTrainer
from psifold.model import HRMConfig
from psifold.training import create_optimizer, LRScheduler
from psifold.data import RfamDataset, collate_rna_batch, explore_rfam_data
from psifold.evaluate import (
    StructureEvaluator,
    benchmark_comparison,
    print_prediction_examples,
)
from psifold.utils import RNA_VOCAB, STRUCTURE_VOCAB


def parse_args():
    parser = argparse.ArgumentParser(description='Train HRM on Rfam dataset')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to Rfam.csv file')
    parser.add_argument('--min_length', type=int, default=10,
                        help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--explore_only', action='store_true',
                        help='Only explore data, don\'t train')

    # Model
    parser.add_argument('--dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_cycles', type=int, default=2,
                        help='Number of high-level cycles (N)')
    parser.add_argument('--cycle_steps', type=int, default=3,
                        help='Low-level steps per cycle (T)')
    parser.add_argument('--l_layers', type=int, default=2,
                        help='L-module Transformer layers')
    parser.add_argument('--h_layers', type=int, default=2,
                        help='H-module Transformer layers')
    parser.add_argument('--use_stablemax', action='store_true',
                        help='Use stablemax instead of softmax')
    parser.add_argument('--use_act', action='store_true', default=True,
                        help='Use Adaptive Computational Time')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Warmup steps')
    parser.add_argument('--max_segments', type=int, default=8,
                        help='Maximum segments for ACT')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Exploration rate for ACT')

    # Data split
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='Training set fraction')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Validation set fraction')
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='Test set fraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    return parser.parse_args()


def train_epoch(trainer, dataloader, scheduler, epoch):
    """Train for one epoch."""
    trainer.model.train()

    total_loss = 0.0
    total_seq_loss = 0.0
    total_q_loss = 0.0
    total_segments = 0.0
    n_batches = 0

    for batch_idx, (sequences, structures, lengths) in enumerate(dataloader):
        # Train step
        metrics = trainer.train_step(
            sequences,
            structures,
            ignore_index=STRUCTURE_VOCAB['<PAD>'],
        )

        # Update learning rate
        scheduler.step()

        # Accumulate metrics
        total_loss += metrics['loss']
        total_seq_loss += metrics['seq_loss']
        total_q_loss += metrics['q_loss']
        total_segments += metrics['avg_segments']
        n_batches += 1

        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"Loss={metrics['loss']:.4f}, "
                  f"SeqLoss={metrics['seq_loss']:.4f}, "
                  f"AvgSegments={metrics['avg_segments']:.2f}")

    return {
        'loss': total_loss / n_batches,
        'seq_loss': total_seq_loss / n_batches,
        'q_loss': total_q_loss / n_batches,
        'avg_segments': total_segments / n_batches,
    }


def eval_epoch(trainer, dataloader):
    """Evaluate for one epoch."""
    trainer.model.eval()
    evaluator = StructureEvaluator()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for sequences, structures, lengths in dataloader:
            # Get predictions
            sequences = sequences.to(trainer.device)
            structures = structures.to(trainer.device)

            # Initialize states
            z_L, z_H = trainer.model.get_initial_state(
                sequences.shape[0],
                sequences.shape[1]
            )

            # Forward pass
            y_hat, (z_L, z_H), _ = trainer.model(
                sequences,
                z_L=z_L,
                z_H=z_H,
                return_q_values=False,
            )

            # Get predictions
            pred_tokens = y_hat.argmax(dim=-1)

            # Evaluate
            evaluator.add_batch(pred_tokens, structures, lengths)

            # Compute loss
            loss = trainer.compute_sequence_loss(
                y_hat,
                structures,
                ignore_index=STRUCTURE_VOCAB['<PAD>'],
            )
            total_loss += loss.item()
            n_batches += 1

    metrics = evaluator.compute()
    metrics['loss'] = total_loss / n_batches

    return metrics, evaluator


def main():
    args = parse_args()

    print("=" * 80)
    print("HRM Training on Rfam Dataset")
    print("=" * 80)

    # Explore data if requested
    if args.explore_only:
        explore_rfam_data(args.data_path)
        return

    # Set random seed
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and split data
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    dataset = RfamDataset(
        args.data_path,
        max_length=args.max_length,
        min_length=args.min_length,
    )

    # Print statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total sequences: {stats['total_sequences']:,}")
    print(f"  Mean length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"  Length range: {stats['min_length']:.0f} - {stats['max_length']:.0f}")

    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_rna_batch,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_rna_batch,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_rna_batch,
        num_workers=0,
    )

    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    config = HRMConfig(
        vocab_size=len(RNA_VOCAB),
        output_vocab_size=len(STRUCTURE_VOCAB),
        dim=args.dim,
        n_heads=args.n_heads,
        n_cycles=args.n_cycles,
        cycle_steps=args.cycle_steps,
        l_layers=args.l_layers,
        h_layers=args.h_layers,
        max_seq_len=args.max_length,
        use_stablemax=args.use_stablemax,
        use_act=args.use_act,
    )

    model = config.create_model()
    print(f"\nModel Configuration:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden dim: {args.dim}")
    print(f"  N cycles × T steps: {args.n_cycles} × {args.cycle_steps} = {args.n_cycles * args.cycle_steps}")
    print(f"  L-module layers: {args.l_layers}")
    print(f"  H-module layers: {args.h_layers}")
    print(f"  Use ACT: {args.use_act}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = LRScheduler(optimizer, warmup_steps=args.warmup_steps, base_lr=args.lr)

    # Create trainer
    trainer = HRMTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        max_segments=args.max_segments,
        epsilon=args.epsilon,
        use_act=args.use_act,
    )

    print(f"\nTraining Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Max segments: {args.max_segments}")
    print(f"  Epochs: {args.n_epochs}")

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, args.n_epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.n_epochs} ---")

        # Train
        train_metrics = train_epoch(trainer, train_loader, scheduler, epoch)
        print(f"\nTrain: Loss={train_metrics['loss']:.4f}, "
              f"SeqLoss={train_metrics['seq_loss']:.4f}, "
              f"AvgSegments={train_metrics['avg_segments']:.2f}")

        # Validate
        val_metrics, _ = eval_epoch(trainer, val_loader)
        print(f"Val:   Loss={val_metrics['loss']:.4f}, "
              f"F1={val_metrics['f1']:.4f}, "
              f"Precision={val_metrics['precision']:.4f}, "
              f"Recall={val_metrics['recall']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'config': config,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  → Saved best model (F1={best_val_f1:.4f})")

        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pt'))

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (Val F1={checkpoint['val_f1']:.4f})")

    # Evaluate on test set
    test_metrics, test_evaluator = eval_epoch(trainer, test_loader)

    test_evaluator.print_summary("Test Set")

    # Benchmark comparison
    comparison = benchmark_comparison(test_metrics['f1'])
    print(f"Performance Level: {comparison}\n")

    # Print example predictions
    print("Getting example predictions...")
    sequences, structures, lengths = next(iter(test_loader))
    sequences = sequences.to(trainer.device)
    structures = structures.to(trainer.device)

    z_L, z_H = model.get_initial_state(sequences.shape[0], sequences.shape[1])

    with torch.no_grad():
        y_hat, _, _ = model(sequences, z_L=z_L, z_H=z_H, return_q_values=False)
        pred_tokens = y_hat.argmax(dim=-1)

    # Get sequence strings
    from psifold.utils import decode_rna_sequence
    seq_strings = [decode_rna_sequence(s.cpu().tolist()[:l.item()])
                   for s, l in zip(sequences, lengths)]

    print_prediction_examples(pred_tokens, structures, seq_strings, n_examples=5)

    # Save final results
    results = {
        'test_f1': test_metrics['f1'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_mcc': test_metrics['mcc'],
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'args': vars(args),
    }

    import json
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")
    print(f"Best model saved to {args.output_dir}/best_model.pt")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
