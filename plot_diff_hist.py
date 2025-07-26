#!/usr/bin/env python3
"""
plot_diff_hist.py  –  Histogram of the difference between column 2 and column 1.

Usage
-----
$ python plot_diff_hist.py data.csv

The script expects the first two columns of *data.csv* to contain numeric data
with headers.  It computes (col2 - col1) for every row, then plots a histogram
whose x‑axis is that difference and whose y‑axis is the count (frequency) of
rows falling into each 2‑unit‑wide bucket.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <csvfile>")
        sys.exit(1)

    csv_file = Path(sys.argv[1])
    df = pd.read_csv(csv_file)

    # ---- basic validation --------------------------------------------------
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns")

    x_col, y_col = df.columns[:2]
    diff = df[y_col] - df[x_col]           # difference (col2 − col1)

    # ---- histogram parameters ---------------------------------------------
    bin_width = 2
    bins = np.arange(diff.min(), diff.max() + bin_width, bin_width)

    # ---- plotting ----------------------------------------------------------
    plt.hist(diff, bins=bins, edgecolor="black")
    plt.xlabel(f"{y_col} − {x_col} (bucket width = {bin_width})")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {y_col} − {x_col}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
