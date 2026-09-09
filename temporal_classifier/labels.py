"""Build a closed action vocabulary from REASSEMBLE low-level labels and align it to a time grid."""

import numpy as np

BACKGROUND_LABEL = "background"


def build_vocab(segments_per_demo: list[list[tuple[float, float, str]]]) -> dict[str, int]:
    labels = sorted({text for segs in segments_per_demo for _, _, text in segs})
    vocab = {BACKGROUND_LABEL: 0}
    for i, label in enumerate(labels, start=1):
        vocab[label] = i
    return vocab


def align_labels_to_grid(
    grid: np.ndarray, segments: list[tuple[float, float, str]], vocab: dict[str, int]
) -> np.ndarray:
    label_ids = np.zeros(len(grid), dtype=np.int64)
    for start, end, text in segments:
        mask = (grid >= start) & (grid < end)
        label_ids[mask] = vocab[text]
    return label_ids
