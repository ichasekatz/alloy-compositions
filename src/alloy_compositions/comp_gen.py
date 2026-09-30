"""Grid-sampled composition space generation for N-component alloy systems."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def generate_compositions(
    elements: list[str],
    n_comps: int = 20,
    max_order: int | None = None,
) -> pd.DataFrame:
    """Generate a uniformly sampled composition grid over an N-element alloy space.

    Constructs all subsystems up to ``max_order`` elements from the pool and
    enumerates the integer-grid simplex points at resolution ``1/n_comps``.
    Duplicate rows and rows that do not sum to exactly 1 are removed.

    Args:
        elements (list[str]): Pool of element symbols (e.g. ``["Al", "Co", "Cr"]``).
        n_comps (int): Grid resolution. Each component fraction is a multiple of
            ``1/n_comps``. Defaults to 20 (5 at.% spacing).
        max_order (int | None): Largest subsystem size to include. Defaults to
            ``len(elements)`` (all subsystems).

    Returns:
        DataFrame with one column per element and one row per composition.
        All rows sum to 1.0. Elements absent from a subsystem have value 0.0.

    Raises:
        ValueError: If ``max_order`` exceeds the number of elements or is less than 2.
    """
    elements = sorted(elements)
    if max_order is None:
        max_order = len(elements)
    if max_order < 2:
        raise ValueError(f"max_order must be >= 2, got {max_order}")
    if max_order > len(elements):
        raise ValueError(
            f"max_order ({max_order}) exceeds number of elements ({len(elements)})"
        )

    frames: list[pd.DataFrame] = []
    for order in range(2, max_order + 1):
        for subset in combinations(elements, order):
            grid_points = _simplex_grid(order, n_comps)
            df = pd.DataFrame(grid_points, columns=list(subset))
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=elements)

    result = pd.concat(frames, ignore_index=True).fillna(0.0)
    result = result.reindex(columns=elements, fill_value=0.0)
    result = result.drop_duplicates().reset_index(drop=True)
    return result


def _simplex_grid(n_components: int, n_comps: int) -> np.ndarray:
    """Enumerate integer-grid simplex points for an n-component system.

    Iterates all integer index tuples in ``[0, n_comps]^n_components`` that
    sum to exactly ``n_comps`` and returns the corresponding atomic fractions.

    Args:
        n_components (int): Number of components (simplex dimension + 1).
        n_comps (int): Grid resolution; each fraction is a multiple of
            ``1/n_comps``.

    Returns:
        Array of shape ``(N, n_components)`` containing atomic fractions.
    """
    points: list[list[int]] = []
    for index in np.ndindex(*[n_comps + 1 for _ in range(n_components)]):
        if sum(index) == n_comps:
            points.append(list(index))
    return np.array(points, dtype=float) / n_comps
