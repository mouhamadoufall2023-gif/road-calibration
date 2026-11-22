import numpy as np
from omegaconf import OmegaConf

from calibrate.friction import FrictionModel, braking_distance, braking_errors

CLASSES = ["dry", "fresh_snow", "ice", "melted_snow", "water", "wet"]
EXPERIMENTS = OmegaConf.to_container(OmegaConf.load("configs/friction/experiments.yaml"))


def test_one_hot_prediction_gives_class_moments():
    fm = FrictionModel(EXPERIMENTS, CLASSES)
    m = fm.moments(np.eye(6))
    assert np.allclose(m["mu_hat"], fm.mu_bar)
    assert np.allclose(m["var_classification"], 0)
    assert np.allclose(m["var_total"], (fm.mu_max - fm.mu_min) ** 2 / 12)


def test_variance_matches_monte_carlo_of_the_mixture():
    fm = FrictionModel(EXPERIMENTS, CLASSES)
    p = np.array([[0.5, 0.0, 0.1, 0.0, 0.15, 0.25]])
    rng = np.random.default_rng(0)
    c = rng.choice(6, size=400_000, p=p[0])
    mu = rng.uniform(fm.mu_min[c], fm.mu_max[c])
    m = fm.moments(p)
    assert abs(m["mu_hat"][0] - mu.mean()) < 1e-3
    assert abs(m["var_total"][0] - mu.var()) < 1e-4


def test_balanced_perfect_classifier_matches_paper_mean():
    # Table III: mean mu_hat ~= 0.4205 on the balanced test set
    fm = FrictionModel(EXPERIMENTS, CLASSES)
    assert abs(fm.moments(np.eye(6))["mu_hat"].mean() - 0.4208) < 1e-3


def test_braking():
    assert np.isclose(braking_distance(np.array([0.8]), 25.0)[0], 625 / (2 * 0.8 * 9.81))
    mabe, rbe = braking_errors(np.array([0.5, 0.2]), np.array([0.5, 0.2]), 25.0)
    assert mabe == 0 and rbe == 0
