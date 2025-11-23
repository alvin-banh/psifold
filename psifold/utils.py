"""
Utility functions for RNA structure prediction.
"""

import torch
from typing import List, Tuple, Dict


# RNA nucleotide vocabulary
RNA_VOCAB = {
    'A': 0,
    'U': 1,
    'G': 2,
    'C': 3,
    '<PAD>': 4,
}

# Structure vocabulary (dot-bracket notation)
STRUCTURE_VOCAB = {
    '.': 0,  # Unpaired
    '(': 1,  # Opening pair
    ')': 2,  # Closing pair
    '<PAD>': 3,
}

# Reverse mappings
RNA_VOCAB_REV = {v: k for k, v in RNA_VOCAB.items()}
STRUCTURE_VOCAB_REV = {v: k for k, v in STRUCTURE_VOCAB.items()}


def encode_rna_sequence(sequence: str) -> List[int]:
    """
    Encode RNA sequence to token indices.

    Args:
        sequence: RNA sequence string (e.g., "AUGC")

    Returns:
        List of token indices
    """
    return [RNA_VOCAB[nuc] for nuc in sequence.upper()]


def decode_rna_sequence(tokens: List[int]) -> str:
    """
    Decode token indices to RNA sequence.

    Args:
        tokens: List of token indices

    Returns:
        RNA sequence string
    """
    return ''.join([RNA_VOCAB_REV[tok] for tok in tokens if tok != RNA_VOCAB['<PAD>']])


def encode_structure(structure: str) -> List[int]:
    """
    Encode dot-bracket structure to token indices.

    Args:
        structure: Dot-bracket structure string (e.g., ".(()).")

    Returns:
        List of token indices
    """
    return [STRUCTURE_VOCAB[char] for char in structure]


def decode_structure(tokens: List[int]) -> str:
    """
    Decode token indices to dot-bracket structure.

    Args:
        tokens: List of token indices

    Returns:
        Dot-bracket structure string
    """
    return ''.join([STRUCTURE_VOCAB_REV[tok] for tok in tokens if tok != STRUCTURE_VOCAB['<PAD>']])


def collate_batch(
    batch: List[Tuple[str, str]],
    max_len: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate batch of (sequence, structure) pairs.

    Args:
        batch: List of (sequence, structure) tuples
        max_len: Maximum sequence length (pads/truncates to this length)

    Returns:
        Tuple of (sequences, structures) tensors
    """
    sequences, structures = zip(*batch)

    # Encode
    encoded_seqs = [encode_rna_sequence(seq) for seq in sequences]
    encoded_structs = [encode_structure(struct) for struct in structures]

    # Determine max length
    if max_len is None:
        max_len = max(len(seq) for seq in encoded_seqs)

    # Pad
    padded_seqs = []
    padded_structs = []

    for seq, struct in zip(encoded_seqs, encoded_structs):
        # Truncate if needed
        seq = seq[:max_len]
        struct = struct[:max_len]

        # Pad
        seq_padded = seq + [RNA_VOCAB['<PAD>']] * (max_len - len(seq))
        struct_padded = struct + [STRUCTURE_VOCAB['<PAD>']] * (max_len - len(struct))

        padded_seqs.append(seq_padded)
        padded_structs.append(struct_padded)

    return torch.tensor(padded_seqs), torch.tensor(padded_structs)


def generate_synthetic_data(
    n_samples: int = 1000,
    min_len: int = 10,
    max_len: int = 50,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Generate synthetic RNA sequence-structure pairs for testing.

    This is a simple generator that creates random sequences and
    random valid structures. In practice, you would use real RNA data.

    Args:
        n_samples: Number of samples to generate
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        seed: Random seed

    Returns:
        List of (sequence, structure) tuples
    """
    import random
    random.seed(seed)

    nucleotides = ['A', 'U', 'G', 'C']
    data = []

    for _ in range(n_samples):
        # Random length
        length = random.randint(min_len, max_len)

        # Random sequence
        sequence = ''.join(random.choice(nucleotides) for _ in range(length))

        # Generate valid structure
        structure = generate_random_structure(length)

        data.append((sequence, structure))

    return data


def generate_random_structure(length: int) -> str:
    """
    Generate random valid dot-bracket structure.

    Args:
        length: Length of structure

    Returns:
        Valid dot-bracket structure string
    """
    import random

    # Simple approach: randomly place pairs
    structure = ['.'] * length
    stack = []

    for i in range(length):
        if random.random() < 0.3 and len(stack) > 0:
            # Close a pair
            j = stack.pop()
            structure[j] = '('
            structure[i] = ')'
        elif random.random() < 0.3:
            # Start a potential pair
            stack.append(i)

    return ''.join(structure)


def calculate_structure_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = STRUCTURE_VOCAB['<PAD>'],
) -> float:
    """
    Calculate per-position accuracy for structure prediction.

    Args:
        pred: Predicted structure tokens [batch, seq_len]
        target: Target structure tokens [batch, seq_len]
        ignore_index: Index to ignore in accuracy calculation

    Returns:
        Accuracy (fraction of correct positions)
    """
    # Create mask for valid positions
    mask = (target != ignore_index)

    # Calculate accuracy only on valid positions
    correct = (pred == target) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


def visualize_prediction(
    sequence: str,
    true_structure: str,
    pred_structure: str,
) -> str:
    """
    Create a visualization of sequence and structures for comparison.

    Args:
        sequence: RNA sequence
        true_structure: True structure
        pred_structure: Predicted structure

    Returns:
        Formatted string for visualization
    """
    lines = [
        "Sequence:   " + sequence,
        "True:       " + true_structure,
        "Predicted:  " + pred_structure,
        "Match:      " + ''.join(['✓' if t == p else '✗' for t, p in zip(true_structure, pred_structure)]),
    ]
    return '\n'.join(lines)
