import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hot_detect import detect_hot
from src.eval import evaluate

labeled_scene = pd.read_csv(ROOT / "data" / "labeled_scenes.csv")

# Shrink visual image to match mask size (visual is 17408px, mask is 2176px, so shrink 8x by averaging)
def shrink(visual_href, mask_shape):
    with rasterio.open(visual_href) as src:
        if (src.height, src.width) != mask_shape:
            image = src.read(
                [1, 2, 3],
                out_shape=(3, mask_shape[0], mask_shape[1]),
                resampling=Resampling.average,
            )
        else:
            image = src.read([1, 2, 3])
    return np.transpose(image, (1, 2, 0))


def load_mask(mask_href):
    with rasterio.open(mask_href) as src:
        return src.read(1)


# Save visualization images for Google Drive
def save_visualization(image, pred, mask, item_id, output_dir):
    """Save side-by-side comparison of original, prediction, and ground truth."""
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original RGB
    axes[0].imshow(image)
    axes[0].set_title(f"Original RGB\n{item_id}")
    axes[0].axis('off')
    
    # HOT prediction
    axes[1].imshow(pred, cmap='gray')
    axes[1].set_title("HOT Prediction")
    axes[1].axis('off')
    
    # Ground truth (mask == 2 is cloud)
    axes[2].imshow(mask == 2, cmap='gray')
    axes[2].set_title("Ground Truth (Maxar)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{item_id.replace('/', '_')}.png", dpi=100, bbox_inches='tight')
    plt.close()


# Run HOT detection on each tile
def run_tile(row, save_images=False, image_output_dir=None):
    mask = load_mask(row["mask_href"])
    image = shrink(row["visual_href"], mask.shape)
    pred = detect_hot(image)
    scores = evaluate(pred, mask == 2, mask != 0)
    
    # Save visualization if requested
    if save_images and image_output_dir:
        save_visualization(image, pred, mask, row["item_id"], image_output_dir)
    
    return {"event": row["event"], "item_id": row["item_id"], **scores}


def main(save_images=False):
    results = []
    image_output_dir = ROOT / "results" / "hot_images" if save_images else None
    
    for i, row in labeled_scene.iterrows():
        try:
            results.append(run_tile(row, save_images=save_images, image_output_dir=image_output_dir))
            print(i + 1, "/", len(labeled_scene), row["item_id"], results[-1]["f1"])
        except Exception as e:
            print(i + 1, "/", len(labeled_scene), row["item_id"], "failed:", e)
            results.append({"event": row["event"], "item_id": row["item_id"], "error": str(e)})

    out = ROOT / "results" / "runner_hot_results.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print("saved", out)
    
    if save_images:
        print(f"Images saved to {image_output_dir}")


if __name__ == "__main__":
    # Set save_images=True to save visualization images for Google Drive
    main(save_images=False)
