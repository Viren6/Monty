#!/usr/bin/env python3
"""
plot_fit.py – Plot a two‑column CSV, add best‑fit line, R², and show the
regression equation.

Usage
-----
$ python plot_fit.py data.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


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
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # ---- linear regression -------------------------------------------------
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    r2 = r2_score(y, y_hat)

    # Equation as a nicely formatted string
    eq = f"{y_col} = {slope:.4g} × {x_col} {intercept:+.4g}"
    print(f"Best‑fit line:\n  {eq}")
    print(f"R²        : {r2:.4f}")

    # ---- plotting ----------------------------------------------------------
    plt.scatter(x, y, s=5, label="data")
    x_line = np.linspace(x.min(), x.max(), 200)
    plt.plot(x_line, slope * x_line + intercept,
             linestyle="--", label=f"{eq}\nR² = {r2:.3f}")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
