#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bidirectional_inverse.benchmark import BenchmarkConfig, benchmark_to_csv
from bidirectional_inverse.graphx import draw_q_graph_from_sparse
from bidirectional_inverse.matrix import generate_sparse_adjacency_list
from bidirectional_inverse.plotting import generate_standard_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Run unified bidirectional inverse benchmarks.")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--min-n", type=int, default=200)
    parser.add_argument("--max-n", type=int, default=1000)
    parser.add_argument("--sparsity", choices=["sqrt", "log", "fixed"], default="sqrt")
    parser.add_argument("--fixed-s", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=Path("data/benchmark_results.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--with-gauss", action="store_true")
    parser.add_argument("--graphx-size", type=int, default=0, help="If > 0, also render GraphX visualization for Q = I - M.")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    samples = np.power(10.0, rng.uniform(np.log10(args.min_n), np.log10(args.max_n), size=args.num_samples))
    samples.sort()

    if args.sparsity == "sqrt":
        sparsity_fn = lambda n: int(np.sqrt(n))
    elif args.sparsity == "log":
        sparsity_fn = lambda n: int(max(1, np.log10(n)))
    else:
        sparsity_fn = lambda n: int(args.fixed_s)

    cfg = BenchmarkConfig(output_csv=args.output_csv, show_gauss=args.with_gauss)
    df = benchmark_to_csv(samples=samples, sparsity_fn=sparsity_fn, cfg=cfg)
    if not df.empty:
        generate_standard_plots(df, args.artifacts_dir)
    if args.graphx_size > 0:
        s = int(sparsity_fn(args.graphx_size))
        _, m_col = generate_sparse_adjacency_list(args.graphx_size, s, cfg.q_spectral_radius, seed=args.seed)
        draw_q_graph_from_sparse(m_col, size=args.graphx_size, output_path=args.artifacts_dir / "q_graphx.png")


if __name__ == "__main__":
    main()
