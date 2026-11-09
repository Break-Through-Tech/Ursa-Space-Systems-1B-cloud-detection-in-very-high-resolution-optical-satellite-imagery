"""Deterministic scene splitting and HOT threshold calibration helpers."""

import hashlib

import numpy as np
import pandas as pd


def assign_scene_splits(
    scenes: pd.DataFrame,
    tuning_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """Split scene IDs into tuning and evaluation sets within each event."""
    if not 0.0 < tuning_fraction < 1.0:
        raise ValueError("tuning_fraction must be between 0 and 1")
    required = {"event", "item_id"}
    missing = required - set(scenes.columns)
    if missing:
        raise ValueError(f"Scene table is missing columns: {sorted(missing)}")
    if scenes["item_id"].astype(str).duplicated().any():
        raise ValueError("Scene item_id values must be unique")

    assignments: dict[str, str] = {}
    for event, group in scenes.groupby("event", sort=True, dropna=False):
        item_ids = sorted(group["item_id"].astype(str))
        if len(item_ids) < 2:
            assignments.update({item_id: "evaluation" for item_id in item_ids})
            continue

        event_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{event}".encode("utf-8")).digest()[:8],
            byteorder="big",
        )
        rng = np.random.default_rng(event_seed)
        tuning_count = min(
            len(item_ids) - 1,
            max(1, round(len(item_ids) * tuning_fraction)),
        )
        tuning_ids = set(rng.choice(item_ids, size=tuning_count, replace=False))
        assignments.update(
            {
                item_id: "tuning" if item_id in tuning_ids else "evaluation"
                for item_id in item_ids
            }
        )
    return assignments


def threshold_f1_scores(
    hot_scores: np.ndarray,
    true_mask: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Return F1 for each threshold without repeatedly scanning the full tile."""
    if hot_scores.shape != true_mask.shape or true_mask.shape != valid_mask.shape:
        raise ValueError("HOT scores, truth mask, and valid mask must have equal shapes")

    scores = np.asarray(hot_scores)[valid_mask.astype(bool)].astype(np.float32)
    truth = np.asarray(true_mask)[valid_mask.astype(bool)].astype(bool)
    if scores.size == 0:
        raise ValueError("Cannot calibrate on a scene with no valid mask pixels")

    thresholds = np.asarray(thresholds, dtype=np.float32)
    sorted_scores = np.sort(scores)
    sorted_positive_scores = np.sort(scores[truth])
    predicted_count = scores.size - np.searchsorted(
        sorted_scores, thresholds, side="right"
    )
    true_positive = truth.sum() - np.searchsorted(
        sorted_positive_scores, thresholds, side="right"
    )
    false_positive = predicted_count - true_positive
    false_negative = truth.sum() - true_positive

    denominator = 2 * true_positive + false_positive + false_negative
    return np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(thresholds, dtype=np.float64),
        where=denominator != 0,
    )
