"""
Explore Rfam dataset structure and statistics.

Usage:
    python examples/explore_rfam.py --data_path /path/to/Rfam.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from psifold.data import explore_rfam_data


def parse_args():
    parser = argparse.ArgumentParser(description='Explore Rfam dataset')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to Rfam.csv file')
    return parser.parse_args()


def main():
    args = parse_args()
    explore_rfam_data(args.data_path)


if __name__ == '__main__':
    main()
