"""Composition space generation with alloy_compositions.

Generates a uniformly sampled grid over a multi-component alloy space and
saves the result to a CSV file.
"""

from __future__ import annotations

from pathlib import Path

from alloy_compositions import generate_compositions

elements = ["Al", "Co", "Cr", "Fe", "Ni"]

if __name__ == "__main__":
    # Binary and ternary subsystems at 10 at.% resolution (max_order=3, n_comps=10)
    df = generate_compositions(elements=elements, n_comps=10, max_order=3)
    print(f"Generated {len(df):,} compositions over {len(elements)}-element space")
    print(df.head())

    out = Path("compositions.csv")
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
