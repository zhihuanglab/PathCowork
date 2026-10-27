# Viewport Processor

Processes dense viewport navigation logs into stable, meaningful navigation events.

## Problem

At 10 Hz, one minute can generate ~600 viewport frames. Raw logs are noisy and redundant.

## Inputs

- Viewer navigation stream (`log.jsonl` or equivalent), containing:
  - timestamp
  - viewport rectangle (`x`, `y`, `w`, `h`)
  - zoom/magnification
  - slide/case identifiers

## Processing Goals

- Denoise jittery micro-movements
- Segment continuous browsing into stable windows
- Extract keyframes and key actions
- Convert navigation behavior into tool-use style events

## Outputs

1. `viewport_keyframes.json`
- stable representative windows with timestamps

2. `viewport_events.jsonl`
- unified tool-use style events (`zoom`, `pan`, `inspect`, `peek`)

3. `viewport_revisit_candidates.json`
- revisit candidates for downstream revisit module

## Example Data

- Source: `videos/example_data/brain/log.jsonl`
- Split command:
  - `python3 processors/split_jsonl_to_modalities.py --input videos/example_data/brain/log.jsonl --name brain`
- Output:
  - `processors/viewport/example_data/brain_coordinates.jsonl` (`level == coordinates`, 523 lines)

## Suggested Event Mapping

- broad-field dwell: `inspect`
- high-magnification short-check: `peek`
- explicit zoom delta: `zoom`
- translation of viewport center: `pan`
