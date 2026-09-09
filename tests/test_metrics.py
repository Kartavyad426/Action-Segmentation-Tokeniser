import numpy as np

from temporal_classifier.metrics import f1_at_k, f1_at_k_corpus, get_segments


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


def test_f1_at_k_corpus_differs_from_macro_average_of_per_demo_f1_at_k():
    # Demo A: a single perfectly-matched segment -> per-demo f1 = 1.0 (tp=1, fp=0, fn=0)
    gt_a = np.array([0, 1, 1, 1, 0])
    pred_a = np.array([0, 1, 1, 1, 0])

    # Demo B: 9 ground-truth segments, prediction misses all of them
    # (tp=0, fp=0, fn=9) -> per-demo f1 = 0.0
    gt_b = np.array([1, 0] * 8 + [1])
    pred_b = np.zeros_like(gt_b)

    macro_avg = (f1_at_k(pred_a, gt_a) + f1_at_k(pred_b, gt_b)) / 2
    assert macro_avg == 0.5

    corpus_f1 = f1_at_k_corpus([(pred_a, gt_a), (pred_b, gt_b)])

    # corpus-level aggregation: tp=1, fp=0, fn=9 -> precision=1.0, recall=0.1
    expected = 2 * 1.0 * 0.1 / (1.0 + 0.1)
    assert abs(corpus_f1 - expected) < 1e-9
    assert corpus_f1 != macro_avg  # materially different aggregation convention


def test_f1_at_k_corpus_is_one_for_all_perfect_matches():
    labels_a = np.array([0, 1, 1, 1, 0, 2, 2, 0])
    labels_b = np.array([0, 3, 3, 0])

    score = f1_at_k_corpus([(labels_a, labels_a), (labels_b, labels_b)])

    assert score == 1.0


def test_f1_at_k_corpus_handles_empty_pairs_list():
    assert f1_at_k_corpus([]) == 0.0
