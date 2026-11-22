"""From surface class probabilities to friction and braking distance (Sec. II-C).

Each class c has a friction interval [mu_min_c, mu_max_c]. Given p(c | x):

    mu | x ~ sum_c p(c | x) U(mu_min_c, mu_max_c)                       (3)
    mu_hat(x) = sum_c p(c | x) mu_bar_c                                 (4)
    Var(mu | x) = sum_c p(c | x) (mu_max_c - mu_min_c)^2 / 12           (5)-(6)
                + sum_c p(c | x) (mu_bar_c - mu_hat(x))^2               (7)
    d(mu) = v^2 / (2 mu g)                                              (9)
    Delta d = d(mu_hat) - d(mu_ref),  mu_ref = mu_bar of the true class (10)
"""
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

G = 9.81


class FrictionModel:
    def __init__(self, intervals: Mapping[str, Sequence[float]], class_names: Sequence[str]):
        missing = [c for c in class_names if c not in intervals]
        if missing:
            raise KeyError("No friction interval for classes {}".format(missing))
        bounds = np.array([intervals[c] for c in class_names], dtype=np.float64)
        if (bounds[:, 0] <= 0).any() or (bounds[:, 1] < bounds[:, 0]).any():
            raise ValueError("Intervals must satisfy 0 < mu_min <= mu_max")
        self.class_names = list(class_names)
        self.mu_min, self.mu_max = bounds[:, 0], bounds[:, 1]
        self.mu_bar = bounds.mean(axis=1)
        self.var_within = (self.mu_max - self.mu_min) ** 2 / 12.0

    def moments(self, probs: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-sample mean and variance of mu | x. ``probs``: [N, C]."""
        mu_hat = probs @ self.mu_bar
        var_physical = probs @ self.var_within
        var_classification = (probs * (self.mu_bar[None] - mu_hat[:, None]) ** 2).sum(1)
        return {
            "mu_hat": mu_hat,
            "var_physical": var_physical,
            "var_classification": var_classification,
            "var_total": var_physical + var_classification,
        }

    def reference(self, labels: np.ndarray) -> np.ndarray:
        return self.mu_bar[labels]


def braking_distance(mu: np.ndarray, speed: float, g: float = G) -> np.ndarray:
    return speed ** 2 / (2.0 * mu * g)


def braking_errors(mu_hat: np.ndarray, mu_ref: np.ndarray, speed: float,
                   g: float = G) -> Tuple[float, float]:
    """MABE (m) and RBE (%) of the stopping distance at ``speed`` (m/s)."""
    d_ref = braking_distance(mu_ref, speed, g)
    abs_err = np.abs(braking_distance(mu_hat, speed, g) - d_ref)
    return float(abs_err.mean()), float((abs_err / d_ref).mean() * 100.0)
