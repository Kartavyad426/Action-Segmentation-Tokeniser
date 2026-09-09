import numpy as np

from temporal_classifier.labels import BACKGROUND_LABEL, align_labels_to_grid, build_vocab


def test_build_vocab_includes_background_and_all_seen_labels():
    segments_per_demo = [
        [(0.0, 1.0, "Grasp"), (1.0, 2.0, "Lift")],
        [(0.0, 1.0, "Approach"), (1.0, 2.0, "Align")],
    ]

    vocab = build_vocab(segments_per_demo)

    assert vocab[BACKGROUND_LABEL] == 0
    assert set(vocab.keys()) == {BACKGROUND_LABEL, "Grasp", "Lift", "Approach", "Align"}
    assert len(set(vocab.values())) == len(vocab)  # all ids unique


def test_align_labels_to_grid_defaults_to_background_outside_segments():
    vocab = {BACKGROUND_LABEL: 0, "Grasp": 1, "Lift": 2}
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.5])
    segments = [(0.0, 1.0, "Grasp"), (1.0, 2.0, "Lift")]

    labels = align_labels_to_grid(grid, segments, vocab)

    assert labels.tolist() == [1, 1, 2, 2, 0]
