from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _scatter_by_method(ax, df: pd.DataFrame, xvar: str, yvar: str):
    palette = {
        "Gauss": "blue",
        "Power": "red",
        "Priority": "black",
        "Bidir": "green",
        "Queue": "orange",
        "Recover": "purple",
        "ML": "brown",
    }
    for name, color in palette.items():
        filtered = df[df["Computed"] == name]
        if filtered.empty:
            continue
        ax.scatter(filtered[xvar], filtered[yvar], color=color, alpha=0.25, label=name)
    ax.legend()


def generate_standard_plots(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # plot 1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("FLOPs per Column Read")
    ax.set_xlabel("log10(Number of Columns Requested)")
    ax.set_ylabel("log10(FLOPs)")
    _scatter_by_method(ax, df, "Cols Fetched", "Avg. FLOPs")
    fig.tight_layout()
    fig.savefig(output_dir / "flops_vs_columns.png", dpi=160)
    plt.close(fig)

    # plot 2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Runtime (seconds)")
    ax.set_xlabel("log10(n)")
    ax.set_ylabel("log10(seconds)")
    dfn = df.assign(n=np.log10(df["n"]))
    _scatter_by_method(ax, dfn, "n", "Time")
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_vs_n.png", dpi=160)
    plt.close(fig)

    # plot 3
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Error")
    ax.set_xlabel("log10(n)")
    ax.set_ylabel("log10(|M_ij - u|_inf)")
    dfe = df[df["Avg. Linf Error"] < 0]
    if not dfe.empty:
        dfe = dfe.assign(n=np.log10(dfe["n"]))
        _scatter_by_method(ax, dfe, "n", "Avg. Linf Error")
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_n.png", dpi=160)
    plt.close(fig)
