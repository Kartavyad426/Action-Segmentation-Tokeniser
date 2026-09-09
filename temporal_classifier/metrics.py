"""Segmental F1@k — the standard temporal-action-segmentation evaluation metric."""

import numpy as np


def get_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    segments = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segments.append((int(labels[start]), start, i))
            start = i
    return segments


def segment_counts(
    pred: np.ndarray, gt: np.ndarray, overlap: float = 0.5, background_label: int = 0
) -> tuple[int, int, int]:
    """Raw (tp, fp, fn) segment counts for a single (pred, gt) pair at a given IoU threshold."""
    pred_segs = [s for s in get_segments(pred) if s[0] != background_label]
    gt_segs = [s for s in get_segments(gt) if s[0] != background_label]

    matched = [False] * len(gt_segs)
    tp = 0
    for p_label, p_start, p_end in pred_segs:
        best_iou, best_j = 0.0, -1
        for j, (g_label, g_start, g_end) in enumerate(gt_segs):
            if matched[j] or g_label != p_label:
                continue
            inter = max(0, min(p_end, g_end) - max(p_start, g_start))
            union = max(p_end, g_end) - min(p_start, g_start)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= overlap:
            matched[best_j] = True
            tp += 1

    fp = len(pred_segs) - tp
    fn = len(gt_segs) - tp
    return tp, fp, fn


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1_at_k(pred: np.ndarray, gt: np.ndarray, overlap: float = 0.5, background_label: int = 0) -> float:
    tp, fp, fn = segment_counts(pred, gt, overlap, background_label)
    return _f1_from_counts(tp, fp, fn)


def f1_at_k_corpus(
    pairs: list[tuple[np.ndarray, np.ndarray]], overlap: float = 0.5, background_label: int = 0
) -> float:
    """Corpus-level F1@k: accumulate tp/fp/fn across all (pred, gt) pairs first, then
    compute a single F1 from the totals. This is the standard temporal-action-segmentation
    aggregation convention (as opposed to macro-averaging per-video F1 scores)."""
    total_tp = total_fp = total_fn = 0
    for pred, gt in pairs:
        tp, fp, fn = segment_counts(pred, gt, overlap, background_label)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return _f1_from_counts(total_tp, total_fp, total_fn)
