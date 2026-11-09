"""Score the HOT detector on the holdout scenes selected in the demo notebook.

Run the notebook's threshold-calibration section first. It writes
``results/hot_threshold.json`` with a fixed threshold and scene split. This
runner scores only the held-out evaluation scenes; calibration and excluded
calibration rows remain in the output with explicit statuses.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import evaluate
from src.methods.hot_data import (
    UnsupportedImageError,
    load_cloud_mask,
    load_rgb_at_shape,
)
from src.methods.hot_detect import detect_hot

SCENES_PATH = ROOT / "data" / "labeled_scenes.csv"
CALIBRATION_PATH = ROOT / "results" / "hot_threshold.json"
RESULTS_PATH = ROOT / "results" / "runner_hot_results.csv"


def read_calibration(scenes: pd.DataFrame) -> dict:
    if not CALIBRATION_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CALIBRATION_PATH}. Run the threshold-calibration section "
            "in notebooks/HOT_labeled_scenes_demo.ipynb first."
        )

    with CALIBRATION_PATH.open(encoding="utf-8") as file:
        calibration = json.load(file)
    if calibration.get("schema_version") != 1:
        raise ValueError(f"Unsupported calibration schema in {CALIBRATION_PATH}")

    threshold = calibration.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("Calibration threshold must be a finite number")

    tuning_ids = set(calibration.get("calibration_item_ids", []))
    evaluation_ids = set(calibration.get("evaluation_item_ids", []))
    excluded = calibration.get("calibration_excluded", {})
    if not isinstance(excluded, dict):
        raise ValueError("calibration_excluded must be a mapping of item IDs to reasons")
    excluded_ids = set(excluded)
    if tuning_ids & evaluation_ids or tuning_ids & excluded_ids or evaluation_ids & excluded_ids:
        raise ValueError("Calibration scene splits overlap")

    scene_ids = set(scenes["item_id"].astype(str))
    if tuning_ids | evaluation_ids | excluded_ids != scene_ids:
        raise ValueError("Calibration scene IDs do not match data/labeled_scenes.csv")

    return {
        "threshold": float(threshold),
        "calibration_ids": tuning_ids,
        "evaluation_ids": evaluation_ids,
        "excluded": excluded,
        "tuning_macro_f1": calibration.get("tuning_macro_f1"),
    }


def save_visualization(image, pred, mask, item_id, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"Original RGB\n{item_id}")
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("HOT prediction")
    axes[2].imshow(mask == 2, cmap="gray")
    axes[2].set_title("Ground truth (Maxar)")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    safe_item_id = item_id.replace("/", "_")
    figure.savefig(output_dir / f"{safe_item_id}.png", dpi=100, bbox_inches="tight")
    plt.close(figure)


def run_tile(row, threshold, save_images=False, image_output_dir=None):
    mask = load_cloud_mask(row["mask_href"])
    valid = mask != 0
    if not valid.any():
        raise ValueError("Cloud mask contains no labeled pixels")

    image = load_rgb_at_shape(row["visual_href"], mask.shape)
    prediction = detect_hot(image, threshold=threshold)
    scores = evaluate(prediction, mask == 2, valid)

    if save_images and image_output_dir is not None:
        save_visualization(
            image, prediction, mask, str(row["item_id"]), image_output_dir
        )
    return scores


def main(save_images=False):
    scenes = pd.read_csv(SCENES_PATH, dtype={"item_id": str})
    if scenes["item_id"].duplicated().any():
        raise ValueError("data/labeled_scenes.csv contains duplicate item IDs")
    calibration = read_calibration(scenes)
    threshold = calibration["threshold"]
    image_output_dir = ROOT / "results" / "hot_images" if save_images else None
    results = []

    for index, row in scenes.iterrows():
        item_id = str(row["item_id"])
        base = {
            "event": row["event"],
            "item_id": item_id,
            "split": "",
            "status": "",
            "threshold": threshold,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "error": "",
        }

        if item_id in calibration["calibration_ids"]:
            base["split"] = "tuning"
            base["status"] = "calibration_only"
            results.append(base)
            continue
        if item_id in calibration["excluded"]:
            base["split"] = "tuning"
            base["status"] = "calibration_excluded"
            base["error"] = calibration["excluded"][item_id]
            results.append(base)
            continue

        base["split"] = "evaluation"
        try:
            scores = run_tile(
                row,
                threshold,
                save_images=save_images,
                image_output_dir=image_output_dir,
            )
        except UnsupportedImageError as error:
            base["status"] = "unsupported"
            base["error"] = str(error)
            print(index + 1, "/", len(scenes), item_id, "unsupported:", error)
        except Exception as error:
            base["status"] = "error"
            base["error"] = str(error)
            print(index + 1, "/", len(scenes), item_id, "failed:", error)
        else:
            base["status"] = "scored"
            base.update(scores)
            print(index + 1, "/", len(scenes), item_id, scores["f1"])
        results.append(base)

    output = pd.DataFrame(results)
    if len(output) != len(scenes) or output["item_id"].duplicated().any():
        raise RuntimeError("Runner did not produce exactly one result per input scene")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(RESULTS_PATH, index=False)
    print(f"Threshold: {threshold}")
    print(f"Tuning macro F1: {calibration['tuning_macro_f1']}")
    print("Scene statuses:")
    print(output["status"].value_counts().to_string())
    print(f"Saved {len(output)} scene rows to {RESULTS_PATH}")
    if save_images:
        print(f"Images saved to {image_output_dir}")


if __name__ == "__main__":
    main(save_images=False)
