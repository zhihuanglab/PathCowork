# Mouse Processor

Processes mouse trajectories and interactions as an additional behavioral signal.

## Inputs

- Mouse stream from viewer logs:
  - timestamp
  - pointer position
  - hover state
  - click/drag events
- Viewport timeline for coordinate mapping

## Processing Goals

- Denoise pointer jitter
- Detect hover-intent and candidate region emphasis
- Extract interaction primitives (move, hover, click, drag)
- Provide support signals for revisit detection

## Outputs

1. `mouse_events.jsonl`
- normalized mouse events aligned to slide/viewport coordinates

2. `mouse_highlights.json`
- hover/click-derived candidate salient regions

## Example Data

- Source: `videos/example_data/brain/log.jsonl`
- Split command:
  - `python3 processors/split_jsonl_to_modalities.py --input videos/example_data/brain/log.jsonl --name brain`
- Output:
  - `processors/mouse/example_data/brain_mouse.jsonl` (`level == mouse`, 8380 lines)
