"""Shared evaluation for the Ursa cloud-detection baselines."""
import numpy as np


def evaluate(pred_mask, true_mask, valid=None):
    """Compare a predicted cloud mask to the ground truth.

    pred_mask, true_mask: arrays of the same shape, 1 = cloud, 0 = not cloud.
    valid: optional bool array, False = skip that pixel (unsure or nodata).
    Returns {"precision", "recall", "f1"}.
    """
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    if valid is None:
        valid = np.ones_like(true, dtype=bool)

    tp = np.sum(pred & true & valid)    # called cloud, really cloud
    fp = np.sum(pred & ~true & valid)   # called cloud, really clear
    fn = np.sum(~pred & true & valid)   # missed cloud

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}