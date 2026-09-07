"""Toy 2D datasets and a terminal scatter plot.

Data generation only — nothing here touches `Value` or computes a gradient.
NumPy is used to build the points; what comes back is plain Python lists, so
it feeds straight into `Neuron.__call__`.

    python -m micrograd.data
"""

import numpy as np


def make_moons(n=100, noise=0.1, seed=0):
    """Two interleaving half-circles. Not linearly separable.

    Returns (X, y):
      X  list of n [x1, x2] pairs of floats
      y  list of n labels, -1.0 or +1.0

    Labels are -1/+1 rather than 0/1 to match a `tanh` output, which lives in
    [-1, 1]. A 0/1 target would be unreachable at one end.
    """
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out

    t_out = np.linspace(0, np.pi, n_out)
    outer = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)

    t_in = np.linspace(0, np.pi, n_in)
    inner = np.stack([1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5], axis=1)

    points = np.concatenate([outer, inner])
    points += rng.normal(0.0, noise, points.shape)

    labels = np.concatenate([-np.ones(n_out), np.ones(n_in)])

    order = rng.permutation(n)          # interleave the classes
    points, labels = points[order], labels[order]

    return points.tolist(), labels.tolist()


def ascii_scatter(X, y, width=61, height=21, predict=None):
    """Print the dataset in the terminal. matplotlib is not allowed until Unit 7.

    `o` and `x` are the two classes. If `predict` is given — a callable taking
    [x1, x2] and returning a number — the background is shaded with the sign of
    its output, so you can watch a decision boundary form.
    """
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    x_lo, x_hi = min(xs) - 0.2, max(xs) + 0.2
    y_lo, y_hi = min(ys) - 0.2, max(ys) + 0.2

    def to_col(v):
        return int((v - x_lo) / (x_hi - x_lo) * (width - 1))

    def to_row(v):
        return int((y_hi - v) / (y_hi - y_lo) * (height - 1))

    grid = [[" "] * width for _ in range(height)]

    if predict is not None:
        for r in range(height):
            for c in range(width):
                gx = x_lo + c / (width - 1) * (x_hi - x_lo)
                gy = y_hi - r / (height - 1) * (y_hi - y_lo)
                grid[r][c] = "." if predict([gx, gy]) > 0 else " "

    for (px, py), label in zip(X, y):
        grid[to_row(py)][to_col(px)] = "x" if label > 0 else "o"

    print("+" + "-" * width + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * width + "+")
    print(f" o = -1   x = +1   n = {len(X)}"
          + ("   . = model predicts +1" if predict else ""))


if __name__ == "__main__":
    X, y = make_moons(n=100, noise=0.1, seed=0)
    print(f"X[0] = {X[0]}   y[0] = {y[0]}")
    print(f"{len(X)} points, {sum(1 for v in y if v > 0)} positive, "
          f"{sum(1 for v in y if v < 0)} negative\n")
    ascii_scatter(X, y)
