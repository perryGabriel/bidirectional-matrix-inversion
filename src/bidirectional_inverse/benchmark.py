from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .algorithms import (
    gaussian_inv_dict,
    ml_estimate_dict,
    pow_estimate_dict,
    pow_estimate_epsilon_dict,
    queue_estimate_dict,
    recover_power_series_dict,
)
from .bidirectional import bidir_dict
from .matrix import dict_to_matrix, generate_sparse_adjacency_list, measure
from .types import SparseMatrix


@dataclass(slots=True)
class BenchmarkConfig:
    output_csv: Path
    q_spectral_radius: float = 0.7
    pow_epsilon: float = 1e-5
    pri_epsilon: float = 1e-5
    bid_epsilon: float = 1e-5
    max_iterations: int = 500
    i: int = 1
    show_gauss: bool = False
    do_forward_error: bool = False
    gauss_cutoff: int = 200
    power_series_cutoff: int = 10_000
    absolute_cutoff: int = 13_000_000
    sample_size: int = 30


def test_method(matrix: SparseMatrix, sample: np.ndarray, method: Callable, **kwargs):
    m_inv = {}
    flops = 0.0
    unique_cols = 0.0
    start = time.time()
    for col in sample:
        u, local_flops, cols = method(matrix, int(col), **kwargs)
        flops += float(local_flops) / sample.size
        unique_cols += float(cols) / sample.size
        m_inv[int(col)] = u
    runtime = (time.time() - start) / sample.size
    return m_inv, flops, unique_cols, runtime


def compute_error(m_inv: np.ndarray, n: int, sample: np.ndarray, **kwargs) -> float:
    if "i" in kwargs:
        if kwargs.get("actual_m_inv") is not None:
            diff = kwargs["actual_m_inv"][kwargs["i"], :] - m_inv[kwargs["i"], :]
        else:
            diff = kwargs["actual_m_np"][kwargs["i"], :] @ m_inv - np.identity(n)[kwargs["i"], sample]
    elif kwargs.get("actual_m_inv") is not None:
        diff = kwargs["actual_m_inv"] - m_inv
    else:
        diff = kwargs["actual_m_np"] @ m_inv - np.identity(n)[:, sample]
    return float(measure(diff, np.inf))


def use_method(matrix: SparseMatrix, n: int, sample: np.ndarray, method: Callable, cfg: BenchmarkConfig, **kwargs):
    m_inv, flops, unique_cols, runtime = test_method(matrix, sample, method, **kwargs)
    if n < cfg.power_series_cutoff:
        if kwargs.get("actual_m_inv") is None:
            kwargs["actual_m_np"] = dict_to_matrix(matrix, n)
        dense = dict_to_matrix(m_inv, n)[:, sample]
        inf_error = compute_error(dense, n, sample, **kwargs)
    else:
        inf_error = 1.0
    return flops, inf_error, unique_cols, runtime


def _method_specs(cfg: BenchmarkConfig):
    return [
        ("Gauss", gaussian_inv_dict, lambda n, sample, actual: {"actual_m_inv": actual}, lambda n: cfg.show_gauss and n < cfg.gauss_cutoff),
        (
            "Power",
            pow_estimate_dict,
            lambda n, sample, actual: {
                "max_iterations": cfg.max_iterations,
                "epsilon": cfg.pow_epsilon,
                "actual_m_inv": actual,
                "verbose": 0,
            },
            lambda n: n < cfg.power_series_cutoff,
        ),
        (
            "Priority",
            pow_estimate_epsilon_dict,
            lambda n, sample, actual: {
                "max_iterations": cfg.max_iterations,
                "epsilon": cfg.pri_epsilon,
                "actual_m_inv": actual,
                "verbose": 0,
            },
            lambda n: True,
        ),
        (
            "Bidir",
            bidir_dict,
            lambda n, sample, actual: {
                "matrix_row": None,
                "i": cfg.i,
                "max_iterations": cfg.max_iterations,
                "epsilon": cfg.bid_epsilon,
                "actual_m_inv": actual,
                "verbose": 0,
            },
            lambda n: True,
        ),
        (
            "Queue",
            queue_estimate_dict,
            lambda n, sample, actual: {
                "max_iterations": cfg.max_iterations,
                "epsilon": cfg.pri_epsilon,
                "actual_m_inv": actual,
                "verbose": 0,
            },
            lambda n: True,
        ),
        (
            "Recover",
            recover_power_series_dict,
            lambda n, sample, actual: {
                "max_iterations": cfg.max_iterations,
                "epsilon": cfg.pri_epsilon,
                "actual_m_inv": actual,
                "verbose": 0,
            },
            lambda n: True,
        ),
        (
            "ML",
            ml_estimate_dict,
            lambda n, sample, actual: {
                "max_iterations": cfg.max_iterations,
                "epsilon": cfg.pri_epsilon,
                "actual_m_inv": actual,
                "verbose": 0,
            },
            lambda n: True,
        ),
    ]


def run_all_methods(
    m_row: SparseMatrix,
    m_col: SparseMatrix,
    n: int,
    cfg: BenchmarkConfig,
    timeout_seconds: float | None = None,
    disable_on_timeout: bool = False,
    disabled_methods: set[str] | None = None,
):
    if disabled_methods is None:
        disabled_methods = set()

    sample_size = min(n, cfg.sample_size)
    sample = np.random.randint(low=0, high=n, size=sample_size, dtype=int)

    if cfg.do_forward_error and n < cfg.power_series_cutoff:
        actual_m_inv = np.linalg.pinv(dict_to_matrix(m_col, n))[:, sample]
    else:
        actual_m_inv = None

    results: dict[str, tuple[float, float, float, float]] = {}
    for name, method, kwargs_builder, should_run in _method_specs(cfg):
        if name in disabled_methods or not should_run(n):
            continue

        kwargs = kwargs_builder(n, sample, actual_m_inv)
        if name == "Bidir":
            kwargs["matrix_row"] = m_row

        start = time.perf_counter()
        results[name] = use_method(m_col, n, sample, method, cfg=cfg, **kwargs)
        elapsed = time.perf_counter() - start

        if timeout_seconds is not None and elapsed > timeout_seconds and disable_on_timeout:
            disabled_methods.add(name)

    return results, disabled_methods


def benchmark_to_csv(
    samples: Iterable[int],
    sparsity_fn: Callable[[int], int],
    cfg: BenchmarkConfig,
    timeout_seconds: float | None = None,
    ordered_samples: bool = True,
    show_progress: bool = True,
):
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if cfg.output_csv.exists():
        df = pd.read_csv(cfg.output_csv)
    else:
        df = pd.DataFrame(columns=["Computed", "n", "s", "Avg. FLOPs", "Cols Fetched", "Avg. Linf Error", "Time"])

    disabled_methods: set[str] = set()
    rows = []
    sample_list = [int(x) for x in samples]
    iterator = tqdm(sample_list, desc="Benchmarking", unit="matrix") if show_progress else sample_list

    try:
        for n in iterator:
            s = int(sparsity_fn(n))
            if n >= cfg.absolute_cutoff:
                continue

            m_row, m_col = generate_sparse_adjacency_list(
                size=n,
                num_out_edges=s,
                sum_of_each_column=cfg.q_spectral_radius,
            )
            results, disabled_methods = run_all_methods(
                m_row,
                m_col,
                n,
                cfg,
                timeout_seconds=timeout_seconds,
                disable_on_timeout=ordered_samples,
                disabled_methods=disabled_methods,
            )

            for method, (flops, err, cols, runtime) in results.items():
                rows.append(
                    {
                        "Computed": method,
                        "n": n,
                        "s": s,
                        "Avg. FLOPs": np.log10(flops),
                        "Cols Fetched": np.log10(cols),
                        "Avg. Linf Error": np.log10(err),
                        "Time": np.log10(runtime),
                    }
                )
    finally:
        if rows:
            df = pd.concat((df, pd.DataFrame(rows)), ignore_index=True)
            df.to_csv(cfg.output_csv, index=False)
    return df
