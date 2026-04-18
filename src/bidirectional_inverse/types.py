from __future__ import annotations

from typing import Dict

SparseVector = Dict[int, float]
SparseMatrix = Dict[int, SparseVector]  # column-major: matrix[col][row]
