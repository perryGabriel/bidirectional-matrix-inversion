# Bidirectional Matrix Inversion

This repository is now structured as a reproducible Python package centered on the **bidirectional inverse algorithm** for sparse matrices.

## Repository layout

- `src/bidirectional_inverse/`: source package.
  - `bidirectional.py`: main bidirectional inverse algorithm (core contribution).
  - `algorithms.py`: all other inverse estimators (Gaussian, power, rounded/priority, queue, recover, ML-feedback, Monte Carlo entry estimate).
  - `ALGORITHMS.md`: plain-English guide comparing each estimator and key differences.
  - `matrix.py`: sparse matrix generation, conversions, norms.
  - `benchmark.py`: unified benchmark/data generation pipeline with append-or-create CSV behavior.
  - `plotting.py`: reproducible figure generation into `artifacts/`.
  - `graphx.py`: graph visualizations for \(Q = I - M\) with NetworkX.
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
python scripts/run_benchmarks.py --num-samples 100 --min-n 1000 --max-n 100000 --sampling-mode ordered --sparsity sqrt --output-csv data/benchmark_results.csv --artifacts-dir artifacts --graphx-size 40 --timeout-seconds 2.0
```

Fixed `n`, varying sparsity `s` (paper-style sweep):

```bash
python scripts/run_benchmarks.py --sweep s --fixed-n 500 --num-samples 80 --min-s 2 --max-s 500 --sampling-mode ordered --output-csv data/benchmark_fixed_n.csv
```

You can use fixed sparsity and include Gaussian baseline:

```bash
python scripts/run_benchmarks.py --sparsity fixed --fixed-s 500 --with-gauss
```

Use random sampling in the same interval:

```bash
python scripts/run_benchmarks.py --sampling-mode random --num-samples 50 --min-n 100 --max-n 100000
```

Notes:
- `--sampling-mode ordered` (default) uses `np.logspace(min_n, max_n, num_samples)` style progression from small to large.
- `--sweep n` (default) varies matrix size. `--sweep s` keeps `n` fixed (default `500`) and varies sparsity.
- `--timeout-seconds` only affects ordered mode: when a method exceeds this runtime on a matrix size, that method is skipped for all larger matrix sizes.
- Benchmarks show a `tqdm` progress bar by default (use `--no-progress` to disable).

## Reproducibility notes

- The benchmark script recreates publication-style comparison figures from generated CSV data.
- Existing legacy CSV and PNG files are preserved in `data/` and `artifacts/`.
- The paper PDF is available in `docs/`.
- The benchmark includes the main bidirectional method and the additional algorithms from legacy notebooks so all approaches can be compared in one output CSV.

## Testing

```bash
pytest
```
