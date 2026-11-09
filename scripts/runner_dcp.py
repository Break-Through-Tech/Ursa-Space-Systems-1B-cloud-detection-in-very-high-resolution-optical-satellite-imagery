"""
Runs the Dark Channel Prior (DCP) cloud detector on every tile in data/labeled_scenes.csv and scores each one against its Maxar cloud mask.

For each row, it loads the cloud mask from mask_href and the image from visual_href. The visual is 17408 px and the mask is 2176 px, so the 
image is shrunk 8x by averaging to match the mask. Both files share the same map projection and bounds, so the pixels line up.

It then runs detect() from src/methods/dcp_detect.py on the shrunk image, using the default settings (threshold t = 0.6, window 15). The 
result is a mask where 1 is cloud.

The prediction is scored with evaluate() from src/eval.py. Truth is mask value 2 (cloud). Value 0 (no data) is ignored. Values 1 (clear) and
 3 (cloud shadow) count as not cloud, so shadow is treated as clear in these results.

The masks come from Maxar's own cloud model, not from people. So these scores measure how closely DCP agrees with Maxar's model, not how 
close it is to the truth.

The output is results/runner_dcp_results.csv, with one row per tile and these columns: event, item_id, precision, recall, f1. If a tile fails
 to load, its row has an error column instead of scores, and the run keeps going. The file is overwritten on every run.

To run it, open a terminal in the repo root and run: python scripts/runner_dcp.py
It needs rasterio, scipy, and pandas in the venv (python -m pip install rasterio scipy pandas).
"""


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.methods.dcp_detect import detect
from src.eval import evaluate

labeled_scene = pd.read_csv(ROOT / "data" / "labeled_scenes.csv")

# What happens to each CSV row:
# shrink it to the mask size if it's too big (since every visual is 17408px and mask is 2176, we will shrink 8X by averageing)
#code: shrink function-> if statement checks if the size is too big -> if big, shrink it by (8x by averaging)
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


# load dcp_detect.py and run it on the visual_href column
# evaluate the score dcp_detect.py outputes using turth is mask == 2, valid is mask !=0.
# Save one row per tile in the result/runner_dcp_results.csv
def run_tile(row):
    mask = load_mask(row["mask_href"])
    image = shrink(row["visual_href"], mask.shape)
    pred = detect(image)
    scores = evaluate(pred, mask == 2, mask != 0)
    return {"event": row["event"], "item_id": row["item_id"], **scores}


# what it saves at the end and where.
# save the metric as it applies detect on each tiles and save the outputs to results (runner_dcp_results)
def main():
    results = []
    for i, row in labeled_scene.iterrows():
        try:
            results.append(run_tile(row))
            print(i + 1, "/", len(labeled_scene), row["item_id"], results[-1]["f1"])
        except Exception as e:
            print(i + 1, "/", len(labeled_scene), row["item_id"], "failed:", e)
            results.append({"event": row["event"], "item_id": row["item_id"], "error": str(e)})

    out = ROOT / "results" / "runner_dcp_results.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print("saved", out)


#Run DCP on the shared tiles across all scenes using the batch runner.
if __name__ == "__main__":
    main()