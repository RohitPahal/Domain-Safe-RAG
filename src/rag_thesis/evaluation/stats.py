"""Uncertainty of per-question metrics: bootstrap confidence intervals and an exact paired test.

The inputs are per-question values (e.g. each question's Recall@10). Paired functions take two
lists aligned by question (same questions, same order), as when two retrievers or two corpora
answer the same questions. Everything is deterministic: resampling uses a fixed seed.

Values that aren't independent can name their cluster (``clusters``, one label per value): the
bootstrap then resamples whole clusters. Example: BioASQ asks one question once per snippet, so
several evaluation qids are the same question; resampling questions, not qids, keeps the interval
honest.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

import numpy as np

N_RESAMPLES = 10_000
CONFIDENCE = 0.95
_BATCH = 1_000  # resamples drawn at a time, so memory stays small for thousands of questions


def bootstrap_ci(
    values: Sequence[float],
    *,
    clusters: Sequence[Hashable] | None = None,
    n_resamples: int = N_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 0,
) -> tuple[float, float, float]:
    """The mean of ``values`` and its percentile bootstrap confidence interval: (mean, low, high).

    With ``clusters``, whole clusters are resampled and each resample's mean is over all the
    values of the drawn clusters; without, every value is its own cluster.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("bootstrap_ci needs a non-empty list of numbers.")
    sums, counts = _cluster_totals(data, clusters)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, _BATCH):
        size = min(_BATCH, n_resamples - start)
        drawn = rng.integers(0, sums.size, size=(size, sums.size))
        means[start : start + size] = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(data.mean()), float(low), float(high)


def _cluster_totals(data: np.ndarray, clusters: Sequence[Hashable] | None) -> tuple[np.ndarray, np.ndarray]:
    """Sum and number of values per cluster (in first-seen order)."""
    if clusters is None:
        return data, np.ones(data.size)
    if len(clusters) != data.size:
        raise ValueError("clusters must give one label per value.")
    index: dict[Hashable, int] = {}
    labels = np.array([index.setdefault(label, len(index)) for label in clusters])
    return np.bincount(labels, weights=data), np.bincount(labels).astype(float)


def paired_bootstrap_ci(
    baseline: Sequence[float],
    other: Sequence[float],
    *,
    clusters: Sequence[Hashable] | None = None,
    n_resamples: int = N_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean per-question difference ``other - baseline`` and its bootstrap CI (questions, or
    ``clusters`` of them, resampled together for both sides)."""
    if len(baseline) != len(other):
        raise ValueError("paired values must cover the same questions.")
    differences = [float(b) - float(a) for a, b in zip(baseline, other, strict=True)]
    return bootstrap_ci(differences, clusters=clusters, n_resamples=n_resamples, confidence=confidence, seed=seed)


def sign_test(gained: int, lost: int) -> float:
    """Exact two-sided sign test (McNemar's exact test) on the discordant questions.

    ``gained``: questions only the other side gets right; ``lost``: only the baseline does.
    Returns the p-value of "neither side is better" (1.0 when no question differs).
    """
    if gained < 0 or lost < 0:
        raise ValueError("counts must be non-negative.")
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1))
    return min(1.0, 2 * tail / 2**n)


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve: the chance that a positive scores higher than a negative (ties count
    half), from average ranks (Mann-Whitney). ``nan`` if there are no positives or no negatives."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    n_pos = sum(bool(label) for label in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):  # tied scores share their average rank
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
