# Bidirectional Matrix Inversion

This repository is now structured as a reproducible Python package centered on the **bidirectional inverse algorithm** for sparse matrices.

## Repository layout

- `src/bidirectional_inverse/`: source package.
  - `algorithms.py`: inverse estimators (Gaussian baseline, power series, rounded/priority, bidirectional).
  - `matrix.py`: sparse matrix generation, conversions, norms.
  - `benchmark.py`: unified benchmark/data generation pipeline with append-or-create CSV behavior.
  - `plotting.py`: reproducible figure generation into `artifacts/`.
  - `experimental.py`: optional Monte Carlo estimator from prior exploration.
- `tests/`: unit tests for package methods.
- `scripts/run_benchmarks.py`: one command to produce benchmark CSV + plots.
- `data/`: benchmark CSV files.
- `artifacts/`: generated and historical figures.
- `docs/`: paper and supplementary documentation.
- `notebooks/`: archived research notebooks.

## Installation

From repository root:

```bash
python -m pip install -e .
```

For development (tests):

```bash
python -m pip install -e .[dev]
```

## Run benchmark pipeline

The script appends to an existing CSV if present, or creates one if missing.

```bash
python scripts/run_benchmarks.py \
  --num-samples 25 \
  --min-n 200 \
  --max-n 5000 \
  --sparsity sqrt \
  --output-csv data/benchmark_results.csv \
  --artifacts-dir artifacts
```

You can use fixed sparsity and include Gaussian baseline:

```bash
python scripts/run_benchmarks.py --sparsity fixed --fixed-s 500 --with-gauss
```

## Reproducibility notes

- The benchmark script recreates publication-style comparison figures from generated CSV data.
- Existing legacy CSV and PNG files are preserved in `data/` and `artifacts/`.
- The paper PDF is available in `docs/`.

## Testing

```bash
pytest
```
