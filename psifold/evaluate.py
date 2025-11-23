"""
Evaluation metrics for RNA 2D structure prediction.

Implements:
- F1 Score (primary metric)
- Precision (PPV)
- Recall (Sensitivity)
- Matthews Correlation Coefficient (MCC)
- Per-structure and per-dataset metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .utils import decode_structure, STRUCTURE_VOCAB


@dataclass
class StructureMetrics:
    """Metrics for a single structure prediction."""
    tp: int  # True positives
    fp: int  # False positives
    tn: int  # True negatives
    fn: int  # False negatives
    precision: float
    recall: float
    f1: float
    mcc: float
    accuracy: float


def base_pairs_from_structure(structure: str) -> set:
    """
    Extract base pairs from dot-bracket structure.

    Args:
        structure: Dot-bracket notation string

    Returns:
        Set of (i, j) tuples where i < j representing base pairs
    """
    pairs = set()
    stack = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.add((j, i))  # (opening, closing) where opening < closing

    return pairs


def calculate_base_pair_metrics(
    pred_structure: str,
    true_structure: str,
) -> StructureMetrics:
    """
    Calculate metrics for base pair prediction.

    Args:
        pred_structure: Predicted dot-bracket structure
        true_structure: True dot-bracket structure

    Returns:
        StructureMetrics object
    """
    # Get base pairs
    pred_pairs = base_pairs_from_structure(pred_structure)
    true_pairs = base_pairs_from_structure(true_structure)

    # Calculate counts
    tp = len(pred_pairs & true_pairs)  # Correct predictions
    fp = len(pred_pairs - true_pairs)  # Incorrect predictions
    fn = len(true_pairs - pred_pairs)  # Missed pairs

    # For RNA, TN is all possible pairs that are correctly predicted as non-paired
    # Total possible pairs for sequence of length n: n*(n-1)/2
    n = len(true_structure)
    total_possible_pairs = n * (n - 1) // 2
    tn = total_possible_pairs - tp - fp - fn

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Matthews Correlation Coefficient
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    # Overall accuracy (correct predictions / total positions)
    accuracy = (tp + tn) / total_possible_pairs if total_possible_pairs > 0 else 0.0

    return StructureMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
        accuracy=accuracy,
    )


def evaluate_predictions(
    pred_tokens: torch.Tensor,
    true_tokens: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Evaluate structure predictions on a batch.

    Args:
        pred_tokens: Predicted structure tokens [batch, seq_len]
        true_tokens: True structure tokens [batch, seq_len]
        lengths: Actual sequence lengths [batch] (optional)

    Returns:
        Dictionary with average metrics across batch
    """
    batch_size = pred_tokens.shape[0]

    all_metrics = []

    for i in range(batch_size):
        # Get tokens for this sequence
        pred_tok = pred_tokens[i].cpu().tolist()
        true_tok = true_tokens[i].cpu().tolist()

        # Get actual length
        if lengths is not None:
            length = lengths[i].item()
            pred_tok = pred_tok[:length]
            true_tok = true_tok[:length]
        else:
            # Find first padding token
            try:
                pad_idx = pred_tok.index(STRUCTURE_VOCAB['<PAD>'])
                pred_tok = pred_tok[:pad_idx]
                true_tok = true_tok[:pad_idx]
            except ValueError:
                pass  # No padding

        # Decode to structures
        pred_struct = decode_structure(pred_tok)
        true_struct = decode_structure(true_tok)

        # Calculate metrics
        metrics = calculate_base_pair_metrics(pred_struct, true_struct)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {
        'precision': np.mean([m.precision for m in all_metrics]),
        'recall': np.mean([m.recall for m in all_metrics]),
        'f1': np.mean([m.f1 for m in all_metrics]),
        'mcc': np.mean([m.mcc for m in all_metrics]),
        'accuracy': np.mean([m.accuracy for m in all_metrics]),
        'tp': np.sum([m.tp for m in all_metrics]),
        'fp': np.sum([m.fp for m in all_metrics]),
        'tn': np.sum([m.tn for m in all_metrics]),
        'fn': np.sum([m.fn for m in all_metrics]),
    }

    return avg_metrics


class StructureEvaluator:
    """Evaluator for tracking metrics across an entire dataset."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.all_metrics = []

    def add_batch(
        self,
        pred_tokens: torch.Tensor,
        true_tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ):
        """Add a batch of predictions."""
        batch_size = pred_tokens.shape[0]

        for i in range(batch_size):
            pred_tok = pred_tokens[i].cpu().tolist()
            true_tok = true_tokens[i].cpu().tolist()

            if lengths is not None:
                length = lengths[i].item()
                pred_tok = pred_tok[:length]
                true_tok = true_tok[:length]
            else:
                try:
                    pad_idx = pred_tok.index(STRUCTURE_VOCAB['<PAD>'])
                    pred_tok = pred_tok[:pad_idx]
                    true_tok = true_tok[:pad_idx]
                except ValueError:
                    pass

            pred_struct = decode_structure(pred_tok)
            true_struct = decode_structure(true_tok)

            metrics = calculate_base_pair_metrics(pred_struct, true_struct)
            self.all_metrics.append(metrics)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if not self.all_metrics:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mcc': 0.0,
                'accuracy': 0.0,
            }

        return {
            'precision': np.mean([m.precision for m in self.all_metrics]),
            'recall': np.mean([m.recall for m in self.all_metrics]),
            'f1': np.mean([m.f1 for m in self.all_metrics]),
            'mcc': np.mean([m.mcc for m in self.all_metrics]),
            'accuracy': np.mean([m.accuracy for m in self.all_metrics]),
            'num_samples': len(self.all_metrics),
        }

    def get_per_sample_f1(self) -> List[float]:
        """Get F1 score for each sample."""
        return [m.f1 for m in self.all_metrics]

    def print_summary(self, name: str = "Evaluation"):
        """Print summary of metrics."""
        metrics = self.compute()

        print(f"\n{'=' * 60}")
        print(f"{name} Results")
        print(f"{'=' * 60}")
        print(f"Samples evaluated: {metrics['num_samples']}")
        print(f"\nBase Pair Prediction Metrics:")
        print(f"  F1 Score:   {metrics['f1']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  MCC:        {metrics['mcc']:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"{'=' * 60}\n")


def benchmark_comparison(f1_score: float) -> str:
    """
    Compare F1 score against known benchmarks.

    Args:
        f1_score: F1 score to compare

    Returns:
        String describing performance level
    """
    if f1_score >= 0.90:
        return "State-of-the-Art (SOTA) - Excellent! 🏆"
    elif f1_score >= 0.85:
        return "Good CNN/Transformer - Competitive ✓"
    elif f1_score >= 0.75:
        return "Simple CNN - Decent baseline"
    elif f1_score >= 0.40:
        return "Baseline (rules) - Needs improvement"
    else:
        return "Below baseline - Check implementation"


def print_prediction_examples(
    pred_tokens: torch.Tensor,
    true_tokens: torch.Tensor,
    sequences: List[str],
    n_examples: int = 3,
):
    """
    Print example predictions for visual inspection.

    Args:
        pred_tokens: Predicted structure tokens [batch, seq_len]
        true_tokens: True structure tokens [batch, seq_len]
        sequences: List of sequence strings
        n_examples: Number of examples to print
    """
    print("\n" + "=" * 80)
    print("Example Predictions")
    print("=" * 80)

    for i in range(min(n_examples, len(sequences))):
        seq = sequences[i]
        pred_tok = pred_tokens[i].cpu().tolist()
        true_tok = true_tokens[i].cpu().tolist()

        # Truncate to sequence length
        pred_tok = pred_tok[:len(seq)]
        true_tok = true_tok[:len(seq)]

        pred_struct = decode_structure(pred_tok)
        true_struct = decode_structure(true_tok)

        # Calculate metrics for this example
        metrics = calculate_base_pair_metrics(pred_struct, true_struct)

        print(f"\nExample {i+1} (Length: {len(seq)}, F1: {metrics.f1:.3f}):")
        print(f"  Sequence:   {seq}")
        print(f"  True:       {true_struct}")
        print(f"  Predicted:  {pred_struct}")

        # Show match/mismatch
        match_str = ''.join(['✓' if t == p else '✗' for t, p in zip(true_struct, pred_struct)])
        print(f"  Match:      {match_str}")

    print("=" * 80 + "\n")


if __name__ == '__main__':
    # Test metrics
    true_struct = "(((...)))"
    pred_struct = "(((...)))"

    metrics = calculate_base_pair_metrics(pred_struct, true_struct)
    print(f"Perfect prediction test:")
    print(f"  F1: {metrics.f1:.3f} (should be 1.0)")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")

    pred_struct = ".((...))."
    metrics = calculate_base_pair_metrics(pred_struct, true_struct)
    print(f"\nPartial prediction test:")
    print(f"  F1: {metrics.f1:.3f}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
