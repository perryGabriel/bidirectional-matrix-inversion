from pathlib import Path

import pandas as pd

from bidirectional_inverse.benchmark import BenchmarkConfig, benchmark_to_csv, run_all_methods
from bidirectional_inverse.matrix import generate_sparse_adjacency_list
from bidirectional_inverse.plotting import generate_standard_plots


def test_run_all_methods_smoke():
    m_row, m_col = generate_sparse_adjacency_list(6, 2, 0.5, seed=2)
    cfg = BenchmarkConfig(output_csv=Path("data/_tmp_test.csv"), max_iterations=20)
    results = run_all_methods(m_row, m_col, 6, cfg)
    assert len(results) == 7


def test_benchmark_appends_or_creates(tmp_path):
    out = tmp_path / "bench.csv"
    cfg = BenchmarkConfig(output_csv=out, max_iterations=20)

    df1 = benchmark_to_csv(samples=[5, 6], sparsity_fn=lambda n: 2, cfg=cfg)
    assert out.exists()
    assert not df1.empty

    before = len(pd.read_csv(out))
    benchmark_to_csv(samples=[7], sparsity_fn=lambda n: 2, cfg=cfg)
    after = len(pd.read_csv(out))
    assert after >= before


def test_plot_generation(tmp_path):
    df = pd.DataFrame(
        [
            {"Computed": "Power", "n": 10, "s": 2, "Avg. FLOPs": 1.0, "Cols Fetched": 1.0, "Avg. Linf Error": -2.0, "Time": -3.0},
            {"Computed": "Priority", "n": 10, "s": 2, "Avg. FLOPs": 0.8, "Cols Fetched": 0.9, "Avg. Linf Error": -2.1, "Time": -3.1},
            {"Computed": "Bidir", "n": 10, "s": 2, "Avg. FLOPs": 0.7, "Cols Fetched": 0.8, "Avg. Linf Error": -1.9, "Time": -3.2},
            {"Computed": "Queue", "n": 10, "s": 2, "Avg. FLOPs": 1.1, "Cols Fetched": 1.2, "Avg. Linf Error": -1.7, "Time": -2.9},
        ]
    )
    out_dir = tmp_path / "artifacts"
    generate_standard_plots(df, out_dir)
    assert (out_dir / "flops_vs_columns.png").exists()
    assert (out_dir / "runtime_vs_n.png").exists()
    assert (out_dir / "error_vs_n.png").exists()
