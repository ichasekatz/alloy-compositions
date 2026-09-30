"""Gaussian perturbation of alloy compositions with non-negativity and renormalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def jitter(
    compositions: pd.DataFrame,
    columns: list[str],
    sigma: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply Gaussian jitter to alloy composition columns.

    Adds independent normal noise to each composition value, clips the result
    to [0, 1] to enforce non-negativity, then renormalizes each row so the
    fractions sum to 1.0.

    Args:
        compositions (pd.DataFrame): Input composition table.
        columns (list[str]): Column names to perturb (e.g. element symbols).
        sigma (float): Standard deviation of the Gaussian noise in at.%.
            Defaults to 0.005 (0.5 at.% perturbation).
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Copy of ``compositions`` with perturbed composition columns. All other
        columns are preserved unchanged.

    Raises:
        ValueError: If any column in ``columns`` is not present in the DataFrame.
    """
    missing = [c for c in columns if c not in compositions.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=compositions[columns].shape)

    perturbed = compositions[columns].values + noise
    perturbed = np.clip(perturbed, 0.0, None)

    row_sums = perturbed.sum(axis=1, keepdims=True)
    perturbed = perturbed / row_sums

    result = compositions.copy()
    result[columns] = perturbed
    return result
