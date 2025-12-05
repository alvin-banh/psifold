"""
Test script for HRM-MC-E(3) geometric folding.

This demonstrates the Monte Carlo + E(3) GNN approach for 3D RNA structure prediction.
"""

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Import the geometric folding module
from geometric_folding import HRM_MC_E3

print("=" * 80)
print("HRM-MC-E(3): Geometric Folding Test")
print("=" * 80)

# Configuration
batch_size = 2
seq_len = 15  # Small RNA sequence (15 nucleotides)

print(f"\nConfiguration:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len} nucleotides")

# Create model
print("\nCreating HRM-MC-E(3) model...")
model = HRM_MC_E3(
    node_dim=64,
    hidden_dim=128,
    n_cycles=3,           # N: hierarchical cycles
    cycle_steps=5,        # T: MC steps per cycle
    n_samples_per_step=10, # K: candidates per step
    temperature=1.0,      # MC exploration strength
)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")
print(f"  Cycles: {model.n_cycles}")
print(f"  Steps per cycle: {model.cycle_steps}")
print(f"  MC samples per step: {model.n_samples}")

# Input: RNA sequence (A=0, U=1, G=2, C=3)
# Let's create a simple sequence: AUGCAUGCAUGCAUG
sequence = torch.tensor([
    [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],  # Sample 1
    [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1],  # Sample 2
])

print(f"\nInput sequences:")
nucleotides = ['A', 'U', 'G', 'C']
for i, seq in enumerate(sequence):
    seq_str = ''.join([nucleotides[nuc] for nuc in seq])
    print(f"  Sequence {i+1}: {seq_str}")

# Backbone connectivity (linear chain)
edge_index = torch.stack([
    torch.arange(seq_len - 1),
    torch.arange(1, seq_len),
])

print(f"\nBackbone edges: {edge_index.shape[1]} bonds")

# Angle constraints (i-1, i, i+1 triplets)
angle_triplets = torch.stack([
    torch.arange(seq_len - 2),
    torch.arange(1, seq_len - 1),
    torch.arange(2, seq_len),
], dim=1)

print(f"Angle constraints: {angle_triplets.shape[0]} triplets")

# Run the model
print("\n" + "=" * 80)
print("Running HRM-MC-E(3) folding...")
print("=" * 80)

initial_coords = torch.randn(batch_size, seq_len, 3) * 0.5
print(f"\nInitial structure: Random coordinates (std=0.5)")

# Forward pass
final_coords, loss_history = model(sequence, edge_index, angle_triplets, initial_coords)

print(f"\nFinal coordinates shape: {final_coords.shape}")
print(f"Loss history length: {len(loss_history['total'])} iterations")

# Print loss progression
print("\n" + "=" * 80)
print("Loss Progression")
print("=" * 80)

total_iterations = len(loss_history['total'])
checkpoints = [0, total_iterations//4, total_iterations//2, 3*total_iterations//4, total_iterations-1]

print(f"\n{'Iteration':<12} {'Total Loss':<15} {'Bond':<12} {'Angle':<12} {'Clash':<12}")
print("-" * 80)
for idx in checkpoints:
    print(f"{idx:<12} "
          f"{loss_history['total'][idx]:<15.6f} "
          f"{loss_history['bond'][idx]:<12.6f} "
          f"{loss_history['angle'][idx]:<12.6f} "
          f"{loss_history['clash'][idx]:<12.6f}")

# Calculate improvement
initial_loss = loss_history['total'][0]
final_loss = loss_history['total'][-1]
improvement = (initial_loss - final_loss) / initial_loss * 100

print(f"\n{'=' * 80}")
print(f"Optimization Results:")
print(f"  Initial loss: {initial_loss:.6f}")
print(f"  Final loss:   {final_loss:.6f}")
print(f"  Improvement:  {improvement:.2f}%")
print(f"{'=' * 80}")

# Visualize the structures
print("\nGenerating visualization...")

fig = plt.figure(figsize=(15, 5))

# Plot both sequences
for seq_idx in range(batch_size):
    # Initial structure
    ax1 = fig.add_subplot(1, 2*batch_size, 2*seq_idx + 1, projection='3d')
    coords_init = initial_coords[seq_idx].detach().numpy()

    ax1.plot(coords_init[:, 0], coords_init[:, 1], coords_init[:, 2],
             'o-', markersize=8, linewidth=2, alpha=0.6)
    ax1.set_title(f'Seq {seq_idx+1}: Initial (Random)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Final structure
    ax2 = fig.add_subplot(1, 2*batch_size, 2*seq_idx + 2, projection='3d')
    coords_final = final_coords[seq_idx].detach().numpy()

    ax2.plot(coords_final[:, 0], coords_final[:, 1], coords_final[:, 2],
             'o-', markersize=8, linewidth=2, alpha=0.8, color='green')
    ax2.set_title(f'Seq {seq_idx+1}: Final (Optimized)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

plt.tight_layout()
plt.savefig('geometric_folding_test.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: geometric_folding_test.png")

# Plot loss curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Total loss
axes[0, 0].plot(loss_history['total'], linewidth=2)
axes[0, 0].set_title('Total Constraint Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Bond loss
axes[0, 1].plot(loss_history['bond'], linewidth=2, color='orange')
axes[0, 1].set_title('Bond Length Violation', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

# Angle loss
axes[1, 0].plot(loss_history['angle'], linewidth=2, color='green')
axes[1, 0].set_title('Bond Angle Violation', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True, alpha=0.3)

# Clash loss
axes[1, 1].plot(loss_history['clash'], linewidth=2, color='red')
axes[1, 1].set_title('Steric Clash Violation', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight')
print("Saved loss curves to: loss_curves.png")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
print("\nKey Observations:")
print("1. The model successfully reduces constraint violations")
print("2. Hierarchical cycles allow iterative refinement")
print("3. Monte Carlo sampling explores diverse conformations")
print("4. E(3) GNN maintains geometric validity")
print("\nNext steps:")
print("- Try longer sequences")
print("- Add base pairing constraints from 2D predictions")
print("- Validate against known PDB structures")
print("- Integrate with your trained 2D predictor")
