# Algorithm Guide (`algorithms.py` and `bidirectional.py`)

This document explains, in plain English, what each inverse estimator in this repository is doing and when to use it.

---

## Big Picture

We want to estimate one column of `M^{-1}` for a sparse matrix `M`.

Most methods here use the identity:

- `M = I - Q`
- `M^{-1} = I + Q + Q^2 + Q^3 + ...` (when `Q` is contractive enough)

So many algorithms are different strategies for exploring and truncating this power-series expansion efficiently on sparse graphs.

---

## Core utilities (shared by many methods)

- `dict_mat_mult`: sparse matrix-vector multiply in dictionary form.
- `mul_q_left`: apply `Q` (implicitly from `M`) to a sparse vector.
- `add_vectors`: merge sparse vectors.

These are not estimators by themselves, but they are the workhorses all estimators rely on.

---

## Main result: Bidirectional (`bidirectional.py`)

### `bidir_dict`

**Idea:** Instead of only expanding from the source side, do a two-sided search:

1. Build a **sink-side in-horizon** from target index `i` (using row-form adjacency).
2. Build a **source-side out-horizon** from column index `j` (using column-form adjacency).
3. Continue expansion but keep flow that remains relevant to the in-horizon.

**Why it helps:** It avoids spending effort on paths that are unlikely to contribute to the target-side support, which can reduce work on large sparse problems.

**Compared with Priority/Power:**
- Power/Priority are one-direction expansions from `j` only.
- Bidir uses structural information from both ends and can prune more aggressively.

---

## Baseline and alternative estimators (`algorithms.py`)

### `gaussian_inv_dict`

A sparse/dictionary adaptation of Gaussian elimination for one inverse column.

- **Pros:** Deterministic and generally accurate baseline.
- **Cons:** Can become very expensive (`~O(n^3)` style behavior in dense worst cases).

Use this mostly as a correctness or small-size reference.

---

### `pow_estimate_dict` (Power series)

Classical truncated Neumann/power-series expansion from `j`:

- Start with `e_j`
- Repeatedly apply `Q`
- Accumulate terms into estimate
- Stop by iteration cap or norm threshold

- **Pros:** Conceptually clean, easy baseline.
- **Cons:** Can grow broad support quickly; expensive on larger problems.

---

### `pow_estimate_epsilon_dict` (Priority/rounded power)

Same core as `pow_estimate_dict`, but rounds/prunes tiny terms at each step.

- **Subtle difference from `pow_estimate_dict`:** pruning happens *during* propagation, not just at the end.
- **Effect:** Often much smaller frontier and lower cost, at potential accuracy tradeoff.

---

### `queue_estimate_dict` (Queue expansion)

Uses a priority queue over frontier flow magnitudes rather than layer-by-layer propagation.

- Expands currently largest-magnitude contribution first.
- Can focus compute on likely-important paths earlier.

Compared to Priority/rounded power:
- Priority method is synchronous by iterations.
- Queue method is asynchronous and best-first by current frontier magnitude.

---

### `recover_power_series_dict` (Recover/pick-up variant)

Power-series style method that tracks tiny remainder terms and attempts to preserve/merge them while iterating.

- **Intent:** avoid throwing away all low-magnitude mass too early.
- **Compared to rounded power:** slightly more conservative with small contributions.

---

### `ml_estimate_dict` (Feedback update)

A heuristic feedback method:

1. Predict `y = M u`
2. Normalize around target component
3. Adjust entries of `u` based on residual structure and diagonal information

- **Pros:** can converge quickly in some structured settings.
- **Cons:** heuristic; stability/quality depends more on matrix characteristics and hyperparameters.

---

### `monte_carlo_estimate_entry`

A path-sampling Monte Carlo estimate for a **single inverse entry**.

- Not a full-column estimator in its current form.
- Useful as an exploratory/interpretability method.

---

## Which algorithms are most similar?

- **Most similar pair:** `pow_estimate_dict` and `pow_estimate_epsilon_dict`.
  - Both are forward Neumann expansions.
  - Difference is pruning strategy during propagation.

- **Related but more structural:** `bidir_dict`.
  - Still expansion-based, but adds target-side horizon information.

- **Different scheduling:** `queue_estimate_dict`.
  - Best-first frontier instead of fixed iteration waves.

- **Different paradigm:** `ml_estimate_dict` and `gaussian_inv_dict`.
  - ML-style residual updates vs elimination.

---

## Practical guidance

- Start with **Priority (`pow_estimate_epsilon_dict`)** for balanced speed/quality.
- Use **Bidirectional (`bidir_dict`)** when structure is large/sparse and target-aware pruning can help.
- Keep **Gaussian** for small-size sanity checks.
- Use **Queue/Recover/ML** as comparative alternatives in benchmarks.
- Use **Monte Carlo entry estimator** for exploratory single-entry analysis, not as your primary full-column benchmark method.
