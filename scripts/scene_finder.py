"""
Description: 

Finds Maxar Open Data tiles that come with a cloud mask and are at least 20% cloud, and lists them in a CSV.

It searches the Maxar Open Data STAC catalog (https://maxar-opendata.s3.amazonaws.com/events/catalog.json) in this order: catalog, then event, then pass, then tiles. A pass is one satellite flyover (one acquisition). If the first tile of a pass has no cloud mask, the whole pass is skipped.

A tile counts as a hit when it has two things: a cloud mask .tif in its assets, and a cloud cover of 20% or more. There are no other requirements.

The masks were made by Maxar's own cloud model, not drawn by people. So scoring DCP against them measures how closely DCP agrees with Maxar's model, not how close it is to the truth.

Mask values: 0 is no data, 1 is clear, 2 is cloud, and 3 is cloud shadow. evaluate() expects 1 = cloud, so later code must convert: cloud is value 2, and value 0 is ignored.

The output is data/labeled_scenes.csv, with one row per tile and these columns: event, item_id, cloud_percent, visual_href, mask_href, ms_href.

To run it, open a terminal in the repo root and run: python scripts/scene_finder.py

The DCP batch runner (scripts/runner_dcp.py) reads this CSV next. It uses two columns: visual_href (the image DCP runs on) and mask_href (the answer key).
"""

import csv
import pystac

CATALOG_URL = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"
MAXAR_CSV = "data/labeled_scenes.csv"


#Check two things: whether the word cloud is in the assets name and whether the file's path ends with ".tiff", 
#meaning it's raster mask rather than the .gpkg outline 
def find_cloud_asset(item):
    for key, asset in item.assets.items():
        if "cloud" in key and asset.href.endswith(".tif"):
            return key, asset.get_absolute_href()
    return None, None


#Scan
def scan(events=None, min_cloud=20):
    rows = []
    catalog = pystac.Catalog.from_file(CATALOG_URL)

    for event in catalog.get_children():
        if events and event.id not in events:
            continue
        print("event:", event.id)

        for acq in event.get_children():
            items = list(acq.get_items())
            if not items:
                continue

            key,_ = find_cloud_asset(items[0])
            if key is None:
                continue

            for item in items:
                key, mask_href = find_cloud_asset(item)
                pct = item.properties.get("tile:clouds_percent", 0)
                if mask_href and pct >= min_cloud:  
                    rows.append({
                        "event": event.id,
                        "item_id": item.id,
                        "cloud_percent": pct,
                        "visual_href": item.assets["visual"].get_absolute_href(),
                        "mask_href": mask_href,
                    })
    return rows

def save(rows, path=MAXAR_CSV):
    if not rows:
        print("no labeled scenes found")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "scenes saved to", path)


MASK_EVENTS = [
    "Cyclone-Chido-Dec15",
    "Hurricane-Maria-PuertoRico-Oct-2017",
    "HurricaneHelene-Oct24",
    "HurricaneMilton-Oct24",
    "India-Floods-Oct-2023",
    "Nepal-Earthquake-Nov-2023",
    "Vanuatu-Earthquake-Dec17",
    "WildFires-LosAngeles-Jan-2025",
    "cyclone-emnati22",
    "ghana-explosion22",
    "southafrica-flooding22",
    "tonga-volcano21",
]

if __name__ == "__main__":
    rows = scan(events=MASK_EVENTS)
    save(rows)