"""evaluation/stats.py: bootstrap confidence intervals and the exact sign test."""

from __future__ import annotations

import pytest

from rag_thesis.evaluation.stats import bootstrap_ci, paired_bootstrap_ci, sign_test


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 12  # 120 questions, mean 0.7
    mean, low, high = bootstrap_ci(values)
    assert mean == pytest.approx(0.7)
    assert low < mean < high
    assert 0.6 < low and high < 0.8  # about +-0.08 for a proportion of 0.7 over 120 questions
    assert bootstrap_ci(values) == (mean, low, high)  # fixed seed
    assert bootstrap_ci(values, seed=1) != (mean, low, high)


def test_constant_values_give_a_zero_width_interval():
    assert bootstrap_ci([0.5] * 40) == (0.5, 0.5, 0.5)


def test_more_questions_give_a_narrower_interval():
    _, low_small, high_small = bootstrap_ci([1, 0] * 50)
    _, low_big, high_big = bootstrap_ci([1, 0] * 2000)
    assert high_big - low_big < (high_small - low_small) / 4


def test_paired_ci_is_about_the_per_question_difference():
    baseline = [1, 1, 0, 1, 0] * 20
    other = [1, 0, 0, 1, 0] * 20  # loses one question in five
    mean, low, high = paired_bootstrap_ci(baseline, other)
    assert mean == pytest.approx(-0.2)
    assert low < -0.2 < high < 0
    with pytest.raises(ValueError, match="same questions"):
        paired_bootstrap_ci([1, 0], [1])


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([])


def test_sign_test_matches_hand_computed_values():
    assert sign_test(0, 0) == 1.0
    assert sign_test(3, 3) == 1.0
    assert sign_test(1, 5) == pytest.approx(2 * (1 + 6) / 64)
    assert sign_test(0, 36) == pytest.approx(2 / 2**36)
    assert sign_test(5, 1) == sign_test(1, 5)  # two-sided
    with pytest.raises(ValueError):
        sign_test(-1, 2)


def test_clusters_of_equal_size_resample_like_their_means():
    # 30 clusters of 3 identical copies: resampling clusters = resampling the 30 cluster means.
    cluster_values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 3
    expanded = [value for value in cluster_values for _ in range(3)]
    labels = [index for index in range(30) for _ in range(3)]
    mean, low, high = bootstrap_ci(expanded, clusters=labels)
    expected = bootstrap_ci(cluster_values)
    assert mean == pytest.approx(expected[0])
    assert (low, high) == pytest.approx(expected[1:])


def test_clusters_widen_the_interval_of_repeated_questions():
    expanded = [value for value in [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 3 for _ in range(3)]
    labels = [index for index in range(30) for _ in range(3)]
    _, low_naive, high_naive = bootstrap_ci(expanded)  # treats the 90 copies as independent
    _, low, high = bootstrap_ci(expanded, clusters=labels)
    assert high - low > 1.5 * (high_naive - low_naive)


def test_unequal_clusters_weight_each_value_once_and_labels_must_match():
    assert bootstrap_ci([1.0, 1.0, 1.0, 0.0], clusters=["a", "a", "a", "b"])[0] == 0.75
    with pytest.raises(ValueError, match="one label per value"):
        bootstrap_ci([1.0, 0.0], clusters=["a"])
    mean, low, high = paired_bootstrap_ci([0, 0, 1, 1], [1, 1, 1, 1], clusters=["q1", "q1", "q2", "q2"])
    assert mean == 0.5 and low == 0.0 and high == 1.0  # only 2 clusters: either both differ or neither


def test_auc_matches_the_pairwise_definition():
    from rag_thesis.evaluation.stats import auc

    assert auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auc([0.5, 0.5], [True, False]) == 0.5  # a tie counts half
    scores, labels = [0.3, 0.7, 0.5, 0.7, 0.1], [False, True, True, False, False]
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    pairs = [(p, n) for p in pos for n in neg]
    assert auc(scores, labels) == pytest.approx(sum((p > n) + 0.5 * (p == n) for p, n in pairs) / len(pairs))
    assert auc([0.1, 0.2], [True, True]) != auc([0.1, 0.2], [True, True])  # nan without negatives
