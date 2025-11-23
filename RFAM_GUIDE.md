# Training HRM on Rfam Dataset - Quick Start Guide

This guide shows you how to train the HRM architecture on your Rfam dataset for RNA 2D structure prediction.

## Prerequisites

Your data folder should contain:
- `Rfam.csv.gz` (13.5 MB) - Primary training data
- Optionally: `RNAsolo.csv.gz`, PDB files for testing

## Step 1: Install Dependencies

```bash
cd psifold
pip install -r requirements.txt
pip install -e .
```

This installs:
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- pandas >= 2.0.0
- einops >= 0.7.0

## Step 2: Explore Your Data

Before training, explore your Rfam dataset to understand its structure:

```bash
python examples/explore_rfam.py --data_path /path/to/your/data/Rfam.csv.gz
```

This will show:
- Total number of sequences
- Length distribution
- RNA family statistics
- Example sequences and structures
- Percentiles for sequence lengths

**Expected Output:**
```
Rfam Dataset Exploration
================================================================================
Loading data from /path/to/Rfam.csv.gz...
Loaded 45000 sequences
...
Dataset Statistics:
  Total sequences: 45,000
  Length range: 15 - 450
  Mean length: 85.3 ± 65.2
  Median length: 68.0
  Number of families: 2,450
```

## Step 3: Train the Model

### Quick Start (Small Model for Testing)

```bash
python examples/train_rfam.py \
  --data_path /path/to/your/data/Rfam.csv.gz \
  --dim 128 \
  --n_epochs 10 \
  --batch_size 16 \
  --max_length 256 \
  --output_dir ./outputs/quick_test
```

**This will:**
- Split data 80/10/10 (train/val/test)
- Train for 10 epochs
- Save best model based on validation F1
- Evaluate on test set
- Print benchmark comparison

**Expected Training Time:**
- On CPU: ~2-4 hours (for 10 epochs, small model)
- On GPU: ~20-40 minutes (for 10 epochs, small model)

### Full Training (Competitive Model)

```bash
python examples/train_rfam.py \
  --data_path /path/to/your/data/Rfam.csv.gz \
  --dim 256 \
  --n_heads 8 \
  --n_cycles 3 \
  --cycle_steps 3 \
  --l_layers 2 \
  --h_layers 2 \
  --n_epochs 50 \
  --batch_size 32 \
  --max_length 512 \
  --max_segments 12 \
  --output_dir ./outputs/full_model
```

**Expected Training Time:**
- On GPU: ~3-6 hours (for 50 epochs)

## Step 4: Monitor Training

The training script prints progress every 50 batches:

```
--- Epoch 5/50 ---
  Batch 50/1800: Loss=0.3245, SeqLoss=0.3100, AvgSegments=4.2
  Batch 100/1800: Loss=0.3012, SeqLoss=0.2890, AvgSegments=4.5
  ...

Train: Loss=0.2850, SeqLoss=0.2720, AvgSegments=4.3
Val:   Loss=0.3120, F1=0.7823, Precision=0.7654, Recall=0.8001
  → Saved best model (F1=0.7823)
```

**Key Metrics to Watch:**
- **F1 Score**: Primary metric (target > 0.75 for good model)
- **AvgSegments**: Should be 4-8 (ACT adapting computation)
- **Loss**: Should decrease over time
- **Val F1 > Train F1**: Model is generalizing well

## Step 5: Evaluate Results

After training completes, you'll see:

```
================================================================================
Test Set Results
================================================================================
Samples evaluated: 4500

Base Pair Prediction Metrics:
  F1 Score:   0.8234
  Precision:  0.8156
  Recall:     0.8315
  MCC:        0.7823
  Accuracy:   0.8901
================================================================================

Performance Level: Good CNN/Transformer - Competitive ✓
```

**Benchmark Comparison:**
- **F1 >= 0.90**: State-of-the-Art (SOTA) 🏆
- **F1 >= 0.85**: Good CNN/Transformer - Competitive ✓
- **F1 >= 0.75**: Simple CNN - Decent baseline
- **F1 >= 0.40**: Baseline (rules)
- **F1 < 0.40**: Below baseline - Check implementation

## Understanding Your Results

### Example Predictions

The script shows example predictions:

```
Example 1 (Length: 45, F1: 0.912):
  Sequence:   GCGGAUUUAGCUCAGDDGGGAGAGCGCCAGACUGAAGAUCUGGAG
  True:       ((((((..((((.......))))(((((.......))))))))))).
  Predicted:  ((((((..((((.......))))(((((.......))))))))))).
  Match:      ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓

Example 2 (Length: 78, F1: 0.745):
  Sequence:   UGCCUGGCGGCCGUAGCGCGGUGGUCCCACCUGACCCCAUGCCGAACUCAGAAGUGAAACGCCGUAGCGCCGAUGGUAG
  True:       (((((((((((....)))))))))))......(((((((.......)))))))...........................
  Predicted:  (((((((((((....)))))))))))......(((((((.......)))))))...(((((.......))))).....
  Match:      ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗
```

### What Good Predictions Look Like

✅ **Good:**
- F1 > 0.80 on most sequences
- Correctly predicts stem regions (paired bases)
- Correctly predicts loop regions (unpaired bases)

❌ **Bad:**
- F1 < 0.50 on many sequences
- Predicts pairs where there should be none (FP)
- Misses actual pairs (FN)

## Hyperparameter Tuning

### Model Architecture

| Parameter | Default | Small | Medium | Large |
|-----------|---------|-------|--------|-------|
| `--dim` | 256 | 128 | 256 | 512 |
| `--n_heads` | 8 | 4 | 8 | 16 |
| `--n_cycles` | 2 | 2 | 3 | 4 |
| `--cycle_steps` | 3 | 2 | 3 | 4 |
| `--l_layers` | 2 | 1 | 2 | 3 |
| `--h_layers` | 2 | 1 | 2 | 3 |

### Training

| Parameter | Default | Fast | Balanced | Thorough |
|-----------|---------|------|----------|----------|
| `--n_epochs` | 20 | 10 | 20 | 50 |
| `--batch_size` | 16 | 32 | 16 | 8 |
| `--lr` | 1e-3 | 5e-3 | 1e-3 | 5e-4 |
| `--max_segments` | 8 | 4 | 8 | 16 |

### Tips

1. **Start small**: Use `--dim 128` and `--n_epochs 10` for quick experiments
2. **GPU memory**: Reduce `--batch_size` if you get OOM errors
3. **Slow convergence**: Increase `--lr` to 5e-3
4. **Overfitting**: Increase `--weight_decay` to 0.1
5. **Need more compute**: Increase `--max_segments` and `--n_cycles`

## Output Files

Training creates:

```
outputs/
├── best_model.pt           # Best model (highest val F1)
├── checkpoint_epoch5.pt    # Checkpoints every 5 epochs
├── checkpoint_epoch10.pt
├── ...
└── results.json           # Final test metrics
```

### Loading Saved Model

```python
import torch
from psifold import HRM, HRMConfig

# Load checkpoint
checkpoint = torch.load('outputs/best_model.pt')

# Create model from saved config
model = checkpoint['config'].create_model()
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation F1: {checkpoint['val_f1']:.4f}")

# Use for inference
model.eval()
# ... your inference code ...
```

## Next Steps

### 1. Test on RNAsolo

Validate on experimental structures:

```bash
python examples/train_rfam.py \
  --data_path /path/to/your/data/RNAsolo.csv.gz \
  --explore_only  # First explore the data
```

Then modify the training script to evaluate your trained model on RNAsolo.

### 2. Test on PDB Structures

Test on specific biological structures:
- `1rna.pdb` - Easy (should get F1 > 0.95)
- `2tpk.pdb` - Medium (target F1 > 0.80)
- `1ebq.pdb` - Hard (target F1 > 0.70)

### 3. Compare to Baselines

Install RNAfold:
```bash
# On Ubuntu/Debian
sudo apt-get install vienna-rna

# On Mac
brew install viennarna
```

Then compare your HRM predictions to RNAfold predictions.

## Troubleshooting

### Error: "FileNotFoundError: Rfam.csv.gz"

Make sure you provide the full path:
```bash
python examples/train_rfam.py --data_path /home/user/data/Rfam.csv.gz
```

### Error: "CUDA out of memory"

Reduce batch size:
```bash
python examples/train_rfam.py ... --batch_size 8
```

Or reduce model size:
```bash
python examples/train_rfam.py ... --dim 128 --max_length 256
```

### Warning: "Removing sequences with invalid nucleotides"

Your CSV might have non-standard nucleotides (N, X, etc.). The data loader automatically filters these out.

### Low F1 Score (< 0.40)

Check:
1. Data format is correct (explore with `explore_rfam.py`)
2. Sequences and structures have matching lengths
3. Train for more epochs (try 50 instead of 10)
4. Use larger model (try `--dim 256`)

### Training is very slow

1. Use GPU: `--device cuda`
2. Increase batch size: `--batch_size 32`
3. Reduce `--max_segments` to 4
4. Disable ACT: Remove `--use_act` flag

## Expected Performance Targets

Based on your dataset documentation:

| Dataset | Baseline | Simple CNN | Your Goal | SOTA |
|---------|----------|------------|-----------|------|
| Rfam test | 0.30 | 0.75 | **0.85** | 0.90 |
| RNAsolo | 0.25 | 0.70 | **0.80** | 0.88 |
| 1rna.pdb | 0.95 | 0.99 | **0.99** | 0.99 |
| 2tpk.pdb | 0.40 | 0.75 | **0.85** | 0.90 |
| 1ebq.pdb | 0.20 | 0.55 | **0.70** | 0.80 |

**Your realistic goal**: Match or exceed the "Your Goal" column.

## Advanced: Inference-Time Scaling

The HRM can use more computation at test time for better accuracy:

```python
# Train with max_segments=8
python examples/train_rfam.py ... --max_segments 8

# At inference, use more segments
trainer.max_segments = 16  # More computation = better accuracy

# Test
metrics = eval_epoch(trainer, test_loader)
print(f"F1 with 16 segments: {metrics['f1']:.4f}")
```

Expected improvement: +2-5% F1 score.

## Questions?

Check:
1. Main README.md for architecture details
2. Code comments in `psifold/` modules
3. Paper on HRM architecture (in README)

Happy training! 🧬🚀
