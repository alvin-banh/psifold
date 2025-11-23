"""
Data loading and preprocessing for RNA structure datasets.

Supports:
- Rfam.csv (primary training data)
- RNAsolo.csv (experimental validation data)
- Custom CSV formats
- Both .csv and .csv.gz files supported
"""

import gzip
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
import numpy as np

from .utils import (
    encode_rna_sequence,
    encode_structure,
    RNA_VOCAB,
    STRUCTURE_VOCAB,
)


class RfamDataset(Dataset):
    """
    Dataset for Rfam RNA sequences and structures.

    Expected CSV columns:
    - sequence: RNA sequence (AUGC letters)
    - structure: Dot-bracket notation
    - family: Rfam family ID (optional)
    - description: Description (optional)
    """

    def __init__(
        self,
        csv_path: str,
        max_length: Optional[int] = 512,
        min_length: Optional[int] = 10,
        filter_families: Optional[List[str]] = None,
    ):
        """
        Args:
            csv_path: Path to CSV file (can be .gz compressed)
            max_length: Maximum sequence length (sequences longer are truncated)
            min_length: Minimum sequence length (shorter sequences are filtered)
            filter_families: Optional list of Rfam families to include
        """
        self.csv_path = csv_path
        self.max_length = max_length
        self.min_length = min_length

        # Load data
        print(f"Loading data from {csv_path}...")
        if csv_path.endswith('.gz'):
            with gzip.open(csv_path, 'rt') as f:
                self.df = pd.read_csv(f)
        else:
            self.df = pd.read_csv(csv_path)

        print(f"Loaded {len(self.df)} sequences")

        # Filter by family if specified
        if filter_families is not None:
            if 'family' in self.df.columns:
                self.df = self.df[self.df['family'].isin(filter_families)]
                print(f"Filtered to {len(self.df)} sequences in families: {filter_families}")

        # Filter by length
        self.df['seq_length'] = self.df['sequence'].str.len()
        original_len = len(self.df)

        if min_length is not None:
            self.df = self.df[self.df['seq_length'] >= min_length]
        if max_length is not None:
            self.df = self.df[self.df['seq_length'] <= max_length]

        print(f"After length filtering ({min_length}-{max_length}): {len(self.df)} sequences")
        print(f"Filtered out: {original_len - len(self.df)} sequences")

        # Reset index
        self.df = self.df.reset_index(drop=True)

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate that sequences and structures are valid."""
        print("Validating data...")

        # Check for missing values
        missing = self.df[['sequence', 'structure']].isnull().sum()
        if missing.any():
            print(f"Warning: Found missing values: {missing}")
            self.df = self.df.dropna(subset=['sequence', 'structure'])

        # Check that sequences only contain valid nucleotides
        valid_nucs = set('AUGC')
        invalid_seqs = []

        for idx, row in self.df.iterrows():
            seq = row['sequence'].upper()
            if not set(seq).issubset(valid_nucs):
                invalid_seqs.append(idx)

        if invalid_seqs:
            print(f"Warning: Removing {len(invalid_seqs)} sequences with invalid nucleotides")
            self.df = self.df.drop(invalid_seqs)
            self.df = self.df.reset_index(drop=True)

        # Check that sequence and structure lengths match
        mismatched = []
        for idx, row in self.df.iterrows():
            if len(row['sequence']) != len(row['structure']):
                mismatched.append(idx)

        if mismatched:
            print(f"Warning: Removing {len(mismatched)} sequences with mismatched lengths")
            self.df = self.df.drop(mismatched)
            self.df = self.df.reset_index(drop=True)

        print(f"Validation complete. Final dataset: {len(self.df)} sequences")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Returns:
            Tuple of (sequence, structure) strings
        """
        row = self.df.iloc[idx]
        return row['sequence'].upper(), row['structure']

    def get_info(self, idx: int) -> Dict:
        """Get full information for a sequence."""
        row = self.df.iloc[idx]
        return row.to_dict()

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_sequences': len(self.df),
            'mean_length': self.df['seq_length'].mean(),
            'median_length': self.df['seq_length'].median(),
            'min_length': self.df['seq_length'].min(),
            'max_length': self.df['seq_length'].max(),
            'std_length': self.df['seq_length'].std(),
        }

        if 'family' in self.df.columns:
            stats['num_families'] = self.df['family'].nunique()
            stats['top_families'] = self.df['family'].value_counts().head(10).to_dict()

        return stats

    def split(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> Tuple['RfamDataset', 'RfamDataset', 'RfamDataset']:
        """
        Split dataset into train/val/test sets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

        # Shuffle
        df_shuffled = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Split
        n = len(df_shuffled)
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)

        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]

        # Create datasets
        train_ds = RfamDataset.__new__(RfamDataset)
        train_ds.df = train_df.reset_index(drop=True)
        train_ds.csv_path = self.csv_path
        train_ds.max_length = self.max_length
        train_ds.min_length = self.min_length

        val_ds = RfamDataset.__new__(RfamDataset)
        val_ds.df = val_df.reset_index(drop=True)
        val_ds.csv_path = self.csv_path
        val_ds.max_length = self.max_length
        val_ds.min_length = self.min_length

        test_ds = RfamDataset.__new__(RfamDataset)
        test_ds.df = test_df.reset_index(drop=True)
        test_ds.csv_path = self.csv_path
        test_ds.max_length = self.max_length
        test_ds.min_length = self.min_length

        print(f"\nDataset split:")
        print(f"  Train: {len(train_ds)} sequences ({train_frac*100:.1f}%)")
        print(f"  Val:   {len(val_ds)} sequences ({val_frac*100:.1f}%)")
        print(f"  Test:  {len(test_ds)} sequences ({test_frac*100:.1f}%)")

        return train_ds, val_ds, test_ds


def collate_rna_batch(
    batch: List[Tuple[str, str]],
    pad_value: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate batch of (sequence, structure) pairs into tensors.

    Args:
        batch: List of (sequence, structure) tuples
        pad_value: Padding value (defaults to vocab padding token)

    Returns:
        Tuple of (sequences, structures, lengths)
        - sequences: [batch, max_len] tensor of sequence tokens
        - structures: [batch, max_len] tensor of structure tokens
        - lengths: [batch] tensor of sequence lengths
    """
    if pad_value is None:
        pad_value = RNA_VOCAB['<PAD>']

    sequences, structures = zip(*batch)

    # Encode
    encoded_seqs = [encode_rna_sequence(seq) for seq in sequences]
    encoded_structs = [encode_structure(struct) for struct in structures]

    # Get lengths
    lengths = torch.tensor([len(seq) for seq in encoded_seqs])
    max_len = max(lengths).item()

    # Pad
    padded_seqs = []
    padded_structs = []

    for seq, struct in zip(encoded_seqs, encoded_structs):
        # Pad
        seq_padded = seq + [RNA_VOCAB['<PAD>']] * (max_len - len(seq))
        struct_padded = struct + [STRUCTURE_VOCAB['<PAD>']] * (max_len - len(struct))

        padded_seqs.append(seq_padded)
        padded_structs.append(struct_padded)

    return (
        torch.tensor(padded_seqs, dtype=torch.long),
        torch.tensor(padded_structs, dtype=torch.long),
        lengths,
    )


def explore_rfam_data(csv_path: str) -> None:
    """
    Explore Rfam dataset and print statistics.

    Args:
        csv_path: Path to Rfam CSV file
    """
    print("=" * 80)
    print("Rfam Dataset Exploration")
    print("=" * 80)

    # Load dataset
    dataset = RfamDataset(csv_path, max_length=None, min_length=None)

    # Print statistics
    stats = dataset.get_statistics()

    print("\nDataset Statistics:")
    print(f"  Total sequences: {stats['total_sequences']:,}")
    print(f"  Length range: {stats['min_length']:.0f} - {stats['max_length']:.0f}")
    print(f"  Mean length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"  Median length: {stats['median_length']:.0f}")

    if 'num_families' in stats:
        print(f"  Number of families: {stats['num_families']}")
        print("\n  Top 10 families:")
        for family, count in stats['top_families'].items():
            print(f"    {family}: {count} sequences")

    # Print examples
    print("\nExample sequences:")
    for i in range(min(3, len(dataset))):
        seq, struct = dataset[i]
        print(f"\n  Example {i+1}:")
        print(f"    Length: {len(seq)}")
        print(f"    Sequence:  {seq[:50]}{'...' if len(seq) > 50 else ''}")
        print(f"    Structure: {struct[:50]}{'...' if len(struct) > 50 else ''}")

    # Length distribution
    print("\nLength distribution:")
    lengths = dataset.df['seq_length']

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(lengths, p)
        print(f"  {p}th percentile: {val:.0f}")

    print("=" * 80)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m psifold.data <path_to_rfam.csv.gz>")
        sys.exit(1)

    explore_rfam_data(sys.argv[1])
