URSA CLOUD DETECTION - DATA FOLDER GUIDE

patches/
  Raw satellite image tiles, unedited.
  Example: belize_r3300_c8500.png (1024x1024 RGB, Maxar, Belize)

masks/
  Everything made from hand labeling a patch:
  - *.json        LabelMe polygons (cloud, clear, unsure, nodata)
  - *_mask.png    Ground truth: white = cloud (1), black = clear (0)
  - *_valid.npy   True = scored, False = unsure/nodata (ignored)
  - *_trace.png   Screenshot of the tracing, for reference

Naming: files share the patch ID (e.g. belize_r3300_c8500) so each mask matches its patch.