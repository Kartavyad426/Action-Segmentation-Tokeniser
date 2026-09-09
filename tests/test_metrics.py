import numpy as np

from temporal_classifier.metrics import f1_at_k, get_segments


def test_get_segments_merges_consecutive_equal_labels():
    labels = np.array([1, 1, 1, 2, 2, 0, 0, 3])

    segments = get_segments(labels)

    assert segments == [(1, 0, 3), (2, 3, 5), (0, 5, 7), (3, 7, 8)]


def test_f1_at_k_is_one_for_a_perfect_match():
    labels = np.array([0, 1, 1, 1, 0, 2, 2, 0])

    score = f1_at_k(labels, labels, overlap=0.5)

    assert score == 1.0


def test_f1_at_k_is_zero_when_no_predicted_segment_matches():
    gt = np.array([0, 1, 1, 1, 0, 2, 2, 0])
    pred = np.array([0, 3, 3, 3, 0, 4, 4, 0])  # entirely different labels

    score = f1_at_k(pred, gt, overlap=0.5)

    assert score == 0.0


def test_f1_at_k_penalizes_low_overlap_below_threshold():
    gt = np.array([1] * 10)
    pred = np.array([1] * 3 + [0] * 7)  # only 30% overlap with the single gt segment

    score = f1_at_k(pred, gt, overlap=0.5)

    assert score == 0.0  # below the 0.5 IoU threshold, counts as a miss
