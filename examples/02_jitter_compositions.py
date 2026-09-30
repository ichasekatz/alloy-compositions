"""Gaussian jitter of alloy compositions with alloy_compositions.

Loads a composition CSV, applies Gaussian noise, renormalizes, and saves the
perturbed dataset. Useful for augmenting training data or exploring composition
neighborhoods.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alloy_compositions import generate_compositions, jitter

elements = ["Al", "Co", "Cr", "Fe", "Ni"]

if __name__ == "__main__":
    # Generate a base composition set
    df = generate_compositions(elements=elements, n_comps=20, max_order=4)
    print(f"Base: {len(df):,} compositions")

    # Jitter with 0.5 at.% noise
    df_jittered = jitter(compositions=df, columns=elements, sigma=0.005, seed=42)
    print(f"Jittered: {len(df_jittered):,} compositions")
    print(df_jittered.head())

    out = Path("compositions_jittered.csv")
    df_jittered.to_csv(out, index=False)
    print(f"Saved to {out}")
