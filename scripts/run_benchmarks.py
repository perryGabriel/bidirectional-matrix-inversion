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
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=100_000)
    parser.add_argument("--sampling-mode", choices=["ordered", "random"], default="ordered")
    parser.add_argument("--sparsity", choices=["sqrt", "log", "fixed"], default="sqrt")
    parser.add_argument("--fixed-s", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="Skip algorithm for larger n after timeout when in ordered mode.")
    parser.add_argument("--output-csv", type=Path, default=Path("data/benchmark_results.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--with-gauss", action="store_true")
    parser.add_argument("--graphx-size", type=int, default=0, help="If > 0, also render GraphX visualization for Q = I - M.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def build_samples(args) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    if args.sampling_mode == "ordered":
        samples = np.logspace(np.log10(args.min_n), np.log10(args.max_n), num=args.num_samples)
        return np.round(samples).astype(int)

    samples = np.power(10.0, rng.uniform(np.log10(args.min_n), np.log10(args.max_n), size=args.num_samples))
    return np.round(samples).astype(int)


def main():
    args = parse_args()
    samples = build_samples(args)

    if args.sparsity == "sqrt":
        sparsity_fn = lambda n: int(np.sqrt(n))
    elif args.sparsity == "log":
        sparsity_fn = lambda n: int(max(1, np.log10(n)))
    else:
        sparsity_fn = lambda n: int(args.fixed_s)

    cfg = BenchmarkConfig(output_csv=args.output_csv, show_gauss=args.with_gauss)
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None
    df = benchmark_to_csv(
        samples=samples,
        sparsity_fn=sparsity_fn,
        cfg=cfg,
        timeout_seconds=timeout,
        ordered_samples=(args.sampling_mode == "ordered"),
        show_progress=not args.no_progress,
    )

    if not df.empty:
        generate_standard_plots(df, args.artifacts_dir)
    if args.graphx_size > 0:
        s = int(sparsity_fn(args.graphx_size))
        _, m_col = generate_sparse_adjacency_list(args.graphx_size, s, cfg.q_spectral_radius, seed=args.seed)
        draw_q_graph_from_sparse(m_col, size=args.graphx_size, output_path=args.artifacts_dir / "q_graphx.png")


if __name__ == "__main__":
    main()
