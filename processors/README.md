# Processors Developer Guide

This README describes the processor-side code added for data reading, quick visualization, and voice transcription.

## What Was Added

1. Modality reader classes with built-in visualization:
- `processors/gaze/GazeData.py`
- `processors/mouse/MouseData.py`
- `processors/viewport/ViewportData.py`
- Shared implementation: `processors/modality_readers.py`

2. One notebook that visualizes all three modalities:
- `processors/gaze/visualization.ipynb`

3. Voice transcription wrapper (no new ASR algorithm):
- `processors/voice/VoiceData.py`
- `processors/voice/transcribe_audio.py`

4. Final output builders:
- `final_outputs/build_outputs.py`
- `final_outputs/README.md`

## Gaze / Mouse / Viewport Usage

### In Python

```python
from processors.gaze.GazeData import GazeData
from processors.mouse.MouseData import MouseData
from processors.viewport.ViewportData import ViewportData

gaze = GazeData("processors/gaze/example_data/brain_eye_movement.jsonl")
mouse = MouseData("processors/mouse/example_data/brain_mouse.jsonl")
viewport = ViewportData("processors/viewport/example_data/brain_coordinates.jsonl")

gaze.visualize(max_points=5000)
mouse.visualize(max_points=5000)
viewport.visualize(max_samples=300, rectangle_stride=15)
```

### Foveated Gaze Attention Contour (From `seg_xxxx_raw.json`)

Use `boxes_final` + `eye` points to render a high-attention area boundary:

```python
from processors.gaze.GazeData import GazeData

GazeData.visualize_attention_contour_from_segment_raw(
    "history/case_003/seg_0010_raw.json",
    grid_size=256,
    attention_tier="low",  # "high" | "mid" | "low"
)
```

Case-level generation is now integrated in:
- `final_outputs/gaze_attention.py`
- `final_outputs/build_outputs.py` via `--eye-attention-tier`

Preset meaning:
- `high`: smaller, stricter ROI
- `mid`: balanced ROI
- `low`: larger ROI

You can still override values manually:

```python
GazeData.visualize_attention_contour_from_segment_raw(
    "history/case_003/seg_0010_raw.json",
    grid_size=256,
    attention_tier="low",
    gaussian_sigma=20.0,
    threshold_quantile=0.75,
)
```

### Notebook Example

Open:
- `processors/gaze/visualization.ipynb`

This notebook loads the example JSONL files and renders all three plots.

## Voice Processing (Online API or Local Whisper)

We do not propose a new voice algorithm. We only standardize outputs from existing backends.

### Example audio

- `processors/voice/example_data/audio.webm`

### Unified output format

Both backends write the same JSON structure:
- `output_format`
- `backend`
- `source_audio`
- `created_at`
- `task`
- `language`
- `duration`
- `text`
- `segments[]` with `id`, `text`, `start`, `end`, `avg_logprob`, `language`, `speaker`, `words`

### CLI examples

Local Whisper:

```bash
python3 processors/voice/transcribe_audio.py \
  --input processors/voice/example_data/audio.webm \
  --output processors/voice/example_data/example_transcript.local.json \
  --backend local_whisper \
  --model large-v3
```

OpenAI API:

```bash
python3 processors/voice/transcribe_audio.py \
  --input processors/voice/example_data/audio.webm \
  --output processors/voice/example_data/example_transcript.openai.json \
  --backend openai_api \
  --model gpt-4o-mini-transcribe
```

## Dependencies

- Required for visualization readers:
  - `matplotlib`
- Optional for voice backends:
  - `openai` (for `openai_api`)
  - `faster-whisper` (for `local_whisper`)
  - `ffmpeg` (typically needed by local audio decoding backends)

## Final Three Outputs

From processed segment dataset (`seg_00xx_*`) you can generate:
- Learning review PDF (`learning_review.pdf`)
- Text-image-grounding dataset (`text_image_grounding.jsonl`)
- QA benchmark from transcript via LLM (`qa_benchmark.json`)

Command:

```bash
python3 final_outputs/build_outputs.py \
  --dataset-root history/case_003 \
  --transcript-json videos/example_data/brain/transcription.json \
  --background-text "brain, suspected GBM, surgical resection" \
  --auto-background \
  --refine-text \
  --output-dir final_outputs/example_data
```

Gemini QA (image + text per segment):

```bash
python3 final_outputs/build_outputs.py \
  --skip-pdf --skip-grounded \
  --qa-provider gemini \
  --qa-model gemini-3-flash-preview \
  --qa-count 25
```

Validation reviewer (yes/no labeling in browser):

```bash
python3 reviewers/build_review_manifest.py
python3 reviewers/launch_reviewer.py
```
