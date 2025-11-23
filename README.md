# Psifold: HRM for RNA 2D Structure Prediction

A PyTorch implementation of **Hierarchical Recurrent Model (HRM)** for RNA 2D structure prediction. This implementation uses multiple RNN layers to differentiate structural patterns through hierarchical processing, temporal separation, and recurrent connectivity.

## Overview

This repository implements the HRM architecture inspired by three fundamental principles of neural computation:

1. **Hierarchical Processing**: Information flows through a hierarchy of modules (L-module and H-module), where higher-level modules integrate information over longer timescales
2. **Temporal Separation**: Different modules operate at distinct timescales through N high-level cycles of T low-level timesteps
3. **Recurrent Connectivity**: Extensive feedback loops enable iterative refinement of predictions

### Key Features

- ✅ **1-Step Gradient Approximation**: O(1) memory instead of O(N×T) using approximate gradients (no BPTT)
- ✅ **Deep Supervision**: Multiple forward passes with periodic parameter updates
- ✅ **Adaptive Computational Time (ACT)**: Q-learning based adaptive halting mechanism
- ✅ **Hierarchical Convergence**: L-module converges within cycles while H-module guides overall computation
- ✅ **Modern Transformer Architecture**: RoPE, GLU, RMSNorm, Post-Norm (Llama-style)
- ✅ **Inference-Time Scaling**: Improved performance with increased compute at inference

## Architecture

### Model Components

```
Input (RNA Sequence)
    ↓
Input Embedding (f_I)
    ↓
┌───────────────────────────────────┐
│  N Cycles × T Steps = N×T total  │
│                                   │
│  For each timestep i:             │
│    z_L^i = f_L(z_L^{i-1}, z_H, x̃) │  ← L-module (low-level)
│                                   │
│    Every T steps (end of cycle):  │
│    z_H^i = f_H(z_H^{i-1}, z_L)    │  ← H-module (high-level)
└───────────────────────────────────┘
    ↓
Output Head (f_O)
    ↓
Predicted Structure
```

### Gradient Approximation

Instead of backpropagating through all N×T timesteps (BPTT), we use a **1-step gradient approximation**:

```python
# All timesteps except the last: NO GRADIENTS
with torch.no_grad():
    for i in range(N * T - 1):
        z_L = L_module(z_L, z_H, x_tilde)
        if (i + 1) % T == 0:
            z_H = H_module(z_H, z_L)

# ONLY the final step has gradients
z_L = L_module(z_L, z_H, x_tilde)
z_H = H_module(z_H, z_L)
y_hat = output_head(z_H)
```

This reduces memory from O(N×T) to O(1) while maintaining good performance.

### Deep Supervision

Training uses multiple forward passes (segments) with gradient detachment between them:

```python
for segment in range(1, M_max + 1):
    # Forward pass
    y_hat, (z_L, z_H), q_values = model(x, z_L, z_H)

    # Compute loss and update
    loss = criterion(y_hat, y_true)
    loss.backward()
    optimizer.step()

    # CRUCIAL: Detach for next segment (1-step approximation)
    z_L = z_L.detach()
    z_H = z_H.detach()
```

### Adaptive Computational Time (ACT)

A Q-learning mechanism learns when to halt computation:

- **Q-head** predicts Q-values for "halt" and "continue" actions
- **Halt** when Q(halt) > Q(continue) and minimum segments reached
- **Exploration**: With probability ε, sample random minimum segments
- **Reward**: Binary reward based on prediction correctness

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/psifold.git
cd psifold

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- einops >= 0.7.0

## Quick Start

### Basic Usage

```python
from psifold import HRM, HRMTrainer
from psifold.model import HRMConfig
from psifold.training import create_optimizer, LRScheduler

# Create model configuration
config = HRMConfig(
    vocab_size=5,           # A, U, G, C + padding
    output_vocab_size=4,    # ., (, ), padding
    dim=256,                # Hidden dimension
    n_heads=8,              # Attention heads
    n_cycles=2,             # N: high-level cycles
    cycle_steps=3,          # T: low-level steps per cycle
    l_layers=2,             # L-module Transformer layers
    h_layers=2,             # H-module Transformer layers
    use_act=True,           # Use Adaptive Computational Time
)

# Create model
model = config.create_model()

# Create optimizer and trainer
optimizer = create_optimizer(model, lr=1e-3, weight_decay=0.01)
trainer = HRMTrainer(
    model=model,
    optimizer=optimizer,
    max_segments=8,
    epsilon=0.1,
)

# Training step
metrics = trainer.train_step(input_sequences, target_structures)
print(f"Loss: {metrics['loss']:.4f}, Avg Segments: {metrics['avg_segments']:.2f}")
```

### Running the Example

```bash
# Train on synthetic RNA data
python examples/train_rna_2d.py
```

This will:
1. Generate synthetic RNA sequence-structure pairs
2. Train HRM with deep supervision and ACT
3. Evaluate on validation set
4. Visualize hierarchical convergence
5. Test inference-time scaling

## Project Structure

```
psifold/
├── README.md
├── requirements.txt
├── setup.py
├── psifold/
│   ├── __init__.py
│   ├── model.py          # HRM model architecture
│   ├── modules.py        # Transformer blocks, L/H modules
│   ├── training.py       # Deep supervision + ACT training
│   └── utils.py          # RNA data utilities
└── examples/
    └── train_rna_2d.py   # Example training script
```

## Key Implementation Details

### 1. Transformer Architecture (Llama-style)

All Transformer blocks follow modern best practices:

- **RoPE**: Rotary Positional Embeddings (better than learned positional encodings)
- **GLU**: Gated Linear Units for feedforward networks (more expressive)
- **RMSNorm**: Root Mean Square Normalization without learnable scale/bias
- **No bias terms**: All linear layers exclude bias for better generalization
- **Post-Norm**: Normalization after residual connections for stability
- **Truncated LeCun Normal Initialization**: For weights (std=1/√dim, truncation=2)

### 2. L-Module and H-Module

Both modules are Transformer-based RNNs that combine inputs via element-wise addition:

```python
# L-module: combines previous L-state, current H-state, and input
z = z_L_prev + z_H_current + x_tilde
for layer in self.layers:
    z = layer(z)  # Transformer block

# H-module: combines previous H-state and final L-state
z = z_H_prev + z_L_final
for layer in self.layers:
    z = layer(z)
```

### 3. Hierarchical Convergence

The model exhibits hierarchical convergence:

- **L-module**: Rapidly converges within each cycle, then "resets" when H-module updates
- **H-module**: Slowly converges across cycles, providing stable high-level guidance
- **Result**: Effective depth of N×T steps while maintaining stability

You can visualize this with `model.forward_verbose()`:

```python
results = model.forward_verbose(input_sequences)
print("L-module residuals:", results['l_residuals'])  # Shows spikes at cycle boundaries
print("H-module residuals:", results['h_residuals'])  # Shows steady convergence
```

### 4. Memory Efficiency

Comparison of memory usage:

| Method | Memory | Computation |
|--------|--------|-------------|
| BPTT | O(N×T) | Full gradients through all timesteps |
| 1-step gradient | O(1) | Only final step gradients |
| DEQ (with Jacobian) | O(1) | Requires Jacobian inversion |

The 1-step gradient is the simplest and most memory-efficient approach.

## Customization

### For Different RNA Tasks

```python
# Modify vocabularies in psifold/utils.py
RNA_VOCAB = {...}           # Input vocabulary
STRUCTURE_VOCAB = {...}     # Output vocabulary

# Adjust model config
config = HRMConfig(
    vocab_size=len(RNA_VOCAB),
    output_vocab_size=len(STRUCTURE_VOCAB),
    # ... other parameters
)
```

### Hyperparameter Tuning

Key hyperparameters to tune:

- `n_cycles` (N): Number of high-level cycles (more = deeper computation)
- `cycle_steps` (T): Steps per cycle (affects convergence within cycles)
- `dim`: Hidden dimension (larger = more capacity, slower)
- `n_heads`: Attention heads (typically 4-16)
- `l_layers` / `h_layers`: Transformer layers per module (typically 1-4)
- `max_segments`: Maximum segments for deep supervision (typically 4-16)
- `epsilon`: Exploration rate for ACT (typically 0.1-0.3)

### Simplified Version (No ACT)

```python
config = HRMConfig(
    use_act=False,  # Disable ACT
    # ... other parameters
)

trainer = HRMTrainer(
    model=model,
    optimizer=optimizer,
    use_act=False,
    max_segments=4,  # Fixed number of segments
)
```

## Performance Tips

1. **Start small**: Use smaller `dim` and fewer layers for initial experiments
2. **Use GPU**: Model is designed for GPU acceleration
3. **Batch size**: Larger batches are more efficient (limited by memory)
4. **Learning rate**: Start with 1e-3 and adjust based on convergence
5. **Warmup**: Use 100-1000 warmup steps for stable training
6. **Inference-time scaling**: Increase `max_segments` at inference for better accuracy

## Understanding the Math

### Hierarchical Dynamics

The model dynamics over N cycles of T steps:

```
z_L^i = f_L(z_L^{i-1}, z_H^{⌊i/T⌋}, x̃; θ_L)   for i = 1, ..., N×T

z_H^k = f_H(z_H^{k-1}, z_L^{k×T}; θ_H)         for k = 1, ..., N
```

### 1-Step Gradient Derivation

At equilibrium, the fixed point satisfies:

```
z_L* = f_L(z_L*, z_H, x̃)
z_H* = f_H(z_H*, z_L*)
```

Using Implicit Function Theorem with Neumann series truncation:

```
∂z_H*/∂θ ≈ ∂f_H/∂θ  (1-step approximation)
```

This is equivalent to backpropagating through only the final update.

### Q-Learning for ACT

The Q-values estimate expected correctness:

```
Q(halt) = P(correct | halt now)
Q(continue) = max{Q(halt)_{t+1}, Q(continue)_{t+1}}
```

Loss: Binary cross-entropy between predicted and target Q-values.

## Citation

This implementation is based on the HRM architecture. If you use this code, please cite the original HRM paper (add citation when available).

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or pull request.

## Future Work

- [ ] Support for real RNA structure datasets (e.g., bpRNA, RNAstrand)
- [ ] Pairing matrix output format (in addition to dot-bracket)
- [ ] Pseudoknot prediction
- [ ] 3D structure prediction
- [ ] Pre-trained models
- [ ] Benchmarking against baselines
- [ ] More sophisticated input combination (e.g., gating instead of addition)
- [ ] Distributed training support

## Contact

For questions or issues, please open a GitHub issue.
