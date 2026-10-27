# Gaze Processor

Processes eye-tracking signals into attention-grounded events aligned with slide context.

## Inputs

- Eye-tracking stream (timestamps + screen or viewport-relative coordinates)
- Viewport timeline from `processors/viewport`
- Optional calibration metadata

## Processing Goals

- Remove invalid/out-of-bound gaze points
- Smooth signal and detect fixations/saccades
- Map gaze to slide coordinates using viewport transforms
- Represent salient attention moments as unified events

## Outputs

1. `gaze_multipolygon.json` (new processor artifact)
- compact gaze attention geometry in two coordinate spaces:
  - `multipolygon_roi_norm` (0..1 normalized in ROI)
  - `multipolygon_source_wsi_px` (integer WSI pixels)
- includes LOD variants for different downstream uses:
  - `multipolygon_lod_roi_norm.training` / `multipolygon_lod_source_wsi_px.training`
  - `multipolygon_lod_roi_norm.visualization` / `multipolygon_lod_source_wsi_px.visualization`
- includes:
  - `roi_rect_source_wsi_px` (`x1,y1,x2,y2`, integer)
  - raw gaze points in both spaces:
    - `points_source_wsi_px`
    - `points_roi_norm`

### Export API (Python)

From `processors/gaze/GazeData.py`:

```python
from processors.gaze.GazeData import GazeData

payload = GazeData.export_segment_raw_multipolygon_json(
    "history/case_003/seg_0010_raw.json",
    output_path="processors/gaze/example_data/seg_0010_gaze_multipolygon.json",
    image_file_name="seg_0010_union.png",
    segment_key=None,
    attention_tier="mid",
)
```

## IVT-IDT Manuscript Rendering

`processors/gaze/GazeData.py` now includes a hybrid fixation filter and manuscript plotting helpers:
- IVT gate (velocity threshold) + IDT refinement (dispersion + minimum duration)
- per-segment comparison render:
  - initial gaze
  - processed gaze
  - initial/processed boundary overlays
- numbered panel render for full dataset or selected rows