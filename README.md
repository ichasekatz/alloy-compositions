<div align="center">

# alloy-compositions

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**Grid-sampled composition space generation and Gaussian perturbation for multi-component alloys.**

</div>

## Overview

`alloy-compositions` provides two lightweight tools for working with N-component alloy composition spaces:

- **`generate_compositions`** — enumerate a uniform simplex grid over any subset of elements up to a chosen order (binary, ternary, …, N-ary)
- **`jitter`** — apply Gaussian noise with clip + renormalization to augment or perturb existing composition datasets

Originally developed as separate repositories (`Composition-Generation`, `Composition-Jitter`); consolidated here with a clean package interface.

## Installation

```bash
git clone https://github.com/ichasekatz/alloy-compositions.git
cd alloy-compositions
uv sync
```

## Quick Start

```python
from alloy_compositions import generate_compositions, jitter

elements = ["Al", "Co", "Cr", "Fe", "Ni"]

# All binary and ternary subsystems at 5 at.% resolution
df = generate_compositions(elements=elements, n_comps=20, max_order=3)
print(f"{len(df):,} compositions")

# Perturb with 0.5 at.% Gaussian noise
df_aug = jitter(df, columns=elements, sigma=0.005, seed=42)
```

## API

### `generate_compositions`

```python
generate_compositions(
    elements: list[str],
    n_comps: int = 20,       # grid resolution (1/n_comps spacing)
    max_order: int | None = None,  # largest subsystem size; default = len(elements)
) -> pd.DataFrame
```

Returns a DataFrame with one column per element. Rows sum to 1.0. Elements absent from a subsystem are 0.0.

### `jitter`

```python
jitter(
    compositions: pd.DataFrame,
    columns: list[str],      # element columns to perturb
    sigma: float = 0.005,    # noise std in at.%
    seed: int = 42,
) -> pd.DataFrame
```

Returns a copy with perturbed composition columns. Non-negativity enforced by clipping; rows renormalized to sum to 1.

## Examples

| Script | Description |
|--------|-------------|
| `examples/01_generate_compositions.py` | Generate and save a composition grid |
| `examples/02_jitter_compositions.py` | Perturb compositions with Gaussian noise |

```bash
uv run python examples/01_generate_compositions.py
```

## Running Tests

```bash
uv run pytest -v
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Citation

If you use this software, please cite:

```bibtex
@software{katz_alloy_compositions_2026,
  author  = {Katz, Chase},
  title   = {alloy-compositions: composition space generation and perturbation},
  year    = {2026},
  url     = {https://github.com/ichasekatz/alloy-compositions},
}
```
