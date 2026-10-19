"""Tercile categories and the ranked probability score, used by 03_forecasts.py and 04_verify.py."""
import numpy as np


def tercile_category(y, lower, upper):
    """0 = below normal (y < lower), 1 = near normal, 2 = above normal (y > upper)."""
    return np.where(y < lower, 0, np.where(y > upper, 2, 1))


def ranked_probability_score(prob, category):
    """RPS of tercile probabilities `prob` (n x 3) for observed categories `category` (n), one value per forecast."""
    cum_prob = np.cumsum(prob, axis=1)[:, :2]
    cum_obs = np.stack([category <= 0, category <= 1], axis=1).astype(float)
    return ((cum_prob - cum_obs) ** 2).sum(axis=1)
